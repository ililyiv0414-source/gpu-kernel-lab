#include "kernel_lab/convolution.hpp"
#include "kernel_lab/cuda_kernels.hpp"
#include "kernel_lab/softmax.hpp"
#include "kernel_lab/validation.hpp"
#include "device_ops.cuh"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <locale>
#include <random>
#include <string>
#include <vector>

namespace {
using namespace kernel_lab;
constexpr unsigned int seed = 20260825U;
volatile float sink = 0;
struct Options {
    std::filesystem::path output = ".";
    int warmup = 3;
    int repeats = 11;
    bool info = false;
    std::string profile;
};
int positive_integer(const std::string& value, bool allow_zero) {
    std::size_t used = 0;
    const int number = std::stoi(value, &used);
    if (used != value.size() || number < (allow_zero ? 0 : 1) || number > 10000)
        throw std::invalid_argument("invalid warmup/repeats (maximum 10000)");
    return number;
}
Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--device-info") { options.info = true; continue; }
        if (i + 1 >= argc) throw std::invalid_argument("missing option value: " + key);
        const std::string value = argv[++i];
        if (key == "--output-dir") options.output = value;
        else if (key == "--warmup") options.warmup = positive_integer(value, true);
        else if (key == "--repeats") options.repeats = positive_integer(value, false);
        else if (key == "--profile-case") {
            if (value != "softmax" && value != "convolution") throw std::invalid_argument("invalid profile case");
            options.profile = value;
        } else throw std::invalid_argument("unknown option: " + key);
    }
    return options;
}
void device_info() {
    int count = 0, runtime = 0, driver = 0;
    device::check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (!count) throw std::runtime_error("no CUDA GPU");
    device::check(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp properties{};
    device::check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    device::check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
    device::check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    std::cout << "logical_device=0\nvisible_devices=" << count
              << "\nname=" << properties.name
              << "\ncompute_capability=" << properties.major << '.' << properties.minor
              << "\nsm_count=" << properties.multiProcessorCount
              << "\nglobal_memory_bytes=" << properties.totalGlobalMem
              << "\nruntime_version=" << runtime << "\ndriver_api_version=" << driver
              << "\ncompiled_cuda_version=" << CUDART_VERSION << '\n';
    // Force context setup outside timing.
    device::Buffer probe(1);
    device::check(cudaDeviceSynchronize(), "initialize context");
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto n = values.size();
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}
template<class Function>
std::vector<double> wall_samples(Function&& function, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) function();
    std::vector<double> samples;
    for (int i = 0; i < repeats; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        function();
        samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
    }
    return samples;
}
template<class Function>
std::vector<double> kernel_samples(Function&& launch, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) launch();
    device::check(cudaDeviceSynchronize(), "warmup synchronization");
    device::Event start, stop;
    std::vector<double> samples;
    for (int i = 0; i < repeats; ++i) {
        device::check(cudaEventRecord(start.get()), "record start");
        launch();
        device::check(cudaEventRecord(stop.get()), "record stop");
        device::check(cudaEventSynchronize(stop.get()), "synchronize event");
        float ms = 0;
        device::check(cudaEventElapsedTime(&ms, start.get(), stop.get()), "elapsed time");
        samples.push_back(ms);
    }
    return samples;
}
class Csv {
public:
    explicit Csv(const std::filesystem::path& directory) {
        const auto summary_path = directory / "gpu_summary.csv";
        const auto raw_path = directory / "gpu_samples.csv";
        if (std::filesystem::exists(summary_path) || std::filesystem::exists(raw_path))
            throw std::runtime_error("refusing to overwrite benchmark CSV; use a new output directory");
        summary.open(summary_path);
        raw.open(raw_path);
        if (!summary || !raw) throw std::runtime_error("output directory must exist and be writable");
        summary.imbue(std::locale::classic());
        raw.imbue(std::locale::classic());
        summary << std::setprecision(12);
        raw << std::setprecision(12);
        summary << "operator,n,cols_or_kernel,implementation,scope,median_ms,min_ms,max_ms,output_elements_per_second,speedup_vs_cpu,max_abs_error,atol,rtol,warmup,repeats,seed,correctness\n";
        raw << "operator,n,cols_or_kernel,implementation,scope,sample,elapsed_ms,seed\n";
    }
    void write(const std::string& op, std::size_t n, std::size_t k,
               const std::string& impl, const std::string& scope,
               const std::vector<double>& samples, std::size_t elements,
               double cpu_ms, double error, const Options& options) {
        for (std::size_t i = 0; i < samples.size(); ++i) {
            if (!std::isfinite(samples[i]) || samples[i] <= 0)
                throw std::runtime_error("non-positive/non-finite timing; no valid performance result");
            raw << op << ',' << n << ',' << k << ',' << impl << ',' << scope << ','
                << i << ',' << samples[i] << ',' << seed << '\n';
        }
        const double ms = median(samples);
        summary << op << ',' << n << ',' << k << ',' << impl << ',' << scope << ','
                << ms << ',' << *std::min_element(samples.begin(), samples.end()) << ','
                << *std::max_element(samples.begin(), samples.end()) << ','
                << double(elements) * 1000 / ms << ',' << cpu_ms / ms << ',' << error << ','
                << (op == "softmax" ? 2e-6 : 2e-4) << ',' << (op == "softmax" ? 2e-5 : 2e-4)
                << ',' << options.warmup << ',' << options.repeats << ',' << seed << ",PASS\n";
    }
    void finish() {
        summary.flush();
        raw.flush();
        if (!summary || !raw) throw std::runtime_error("CSV write failed");
    }
private:
    std::ofstream summary, raw;
};
void run_case(const std::string& op, std::size_t n, std::size_t k, const Options& options, Csv* csv) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-2, 2);
    const bool soft = op == "softmax";
    std::vector<float> input(soft ? n * k : n), weights(soft ? 1 : k);
    for (auto& v : input) v = distribution(generator);
    for (auto& v : weights) v = distribution(generator);
    const auto reference = soft ? softmax_reference(input, n, k) : convolution_direct(input, weights);
    const double atol = soft ? 2e-6 : 2e-4, rtol = soft ? 2e-5 : 2e-4;
    device::Buffer d_input(input.size()), d_weights(weights.size()), d_output(reference.size());
    d_input.upload(input);
    d_weights.upload(weights);
    const auto launch = [&] {
        if (soft) device::launch_softmax(d_input.data(), d_output.data(), n, k);
        else device::launch_convolution(d_input.data(), d_weights.data(), d_output.data(), n, k);
    };
    const auto verify = [&](const std::vector<float>& actual) {
        const double error = require_close(actual, reference, atol, rtol, op);
        if (soft) require_probability_rows(actual, n, k);
        return error;
    };
    if (!options.profile.empty()) {
        // Exactly 3 warmup + 5 launches. Profilers may skip the first 3.
        // No timing CSV is generated under instrumentation.
        for (int i = 0; i < 3; ++i) launch();
        device::check(cudaDeviceSynchronize(), "profile warmup");
        for (int i = 0; i < 5; ++i) launch();
        device::check(cudaDeviceSynchronize(), "profile synchronization");
        verify(d_output.download());
        std::cout << "PROFILE_CASE_PASS " << op << "; no benchmark timings emitted\n";
        return;
    }
    const auto cpu = [&] {
        const auto output = soft ? softmax_reference(input, n, k) : convolution_direct(input, weights);
        sink += output.front();
    };
    const auto end_to_end = [&] {
        return soft ? softmax_cuda_naive(input, n, k) : convolution_cuda_naive(input, weights);
    };
    double error = verify(end_to_end());
    launch();
    device::check(cudaDeviceSynchronize(), "pre-timing correctness");
    error = std::max(error, verify(d_output.download()));
    const auto cpu_times = wall_samples(cpu, options.warmup, options.repeats);
    const auto gpu_times = kernel_samples(launch, options.warmup, options.repeats);
    error = std::max(error, verify(d_output.download()));
    const auto total_times = wall_samples([&] {
        const auto output = end_to_end();
        sink += output.front();
    }, options.warmup, options.repeats);
    error = std::max(error, verify(end_to_end()));
    const double cpu_ms = median(cpu_times);
    csv->write(op, n, k, soft ? "cpu_reference" : "cpu_direct", "cpu_wall",
               cpu_times, reference.size(), cpu_ms, 0, options);
    csv->write(op, n, k, "cuda_naive", "kernel_event", gpu_times, reference.size(), cpu_ms, error, options);
    csv->write(op, n, k, "cuda_naive", "end_to_end_wall", total_times, reference.size(), cpu_ms, error, options);
    std::cout << "BENCHMARK_CASE_PASS " << op << ' ' << n << 'x' << k << '\n';
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        device_info();
        if (options.info) return 0;
        if (!options.profile.empty()) {
            run_case(options.profile, options.profile == "softmax" ? 1024 : 32768,
                     options.profile == "softmax" ? 1024 : 255, options, nullptr);
            return 0;
        }
        Csv csv(options.output);
        for (std::size_t cols : {128U, 512U, 1024U}) run_case("softmax", 1024, cols, options, &csv);
        for (std::size_t k : {15U, 63U, 255U}) run_case("convolution", 32768, k, options, &csv);
        csv.finish();
        std::cout << "BENCHMARK_PASS 18 summary rows; sink=" << sink << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
