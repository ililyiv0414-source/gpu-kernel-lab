#include "kernel_lab/cuda_kernels.hpp"
#include "kernel_lab/softmax.hpp"
#include "kernel_lab/softmax_workspace.hpp"
#include "kernel_lab/validation.hpp"
#include "device_ops.cuh"
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <random>

namespace {
using namespace kernel_lab;
using Clock = std::chrono::steady_clock;
constexpr int repeats = 21, warmup = 5;
constexpr unsigned seed = 20260829U;
const char* names[] = {"warp_wrapper", "workspace_fresh", "workspace_reuse"};
double milliseconds(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end()); return values[values.size()/2];
}
std::vector<float> fresh(const std::vector<float>& input, std::size_t rows, std::size_t cols) {
    SoftmaxWorkspace temporary;
    return temporary.run(input, rows, cols); // includes destruction before caller returns
}
std::ofstream open_csv(const std::filesystem::path& directory, const char* filename) {
    const auto path = directory / filename;
    if (std::filesystem::exists(path)) throw std::runtime_error("refusing to overwrite CSV");
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("output directory missing/unwritable");
    stream.imbue(std::locale::classic()); stream << std::setprecision(12);
    return stream;
}
void run_case(std::size_t rows, std::size_t cols, std::ofstream& summary,
              std::ofstream& samples, std::ofstream& stages, std::ofstream& setups) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-2, 2);
    std::vector<float> input(rows * cols);
    for (auto& v : input) v = distribution(generator);
    const auto reference = softmax_reference(input, rows, cols);
    std::array<double, 3> errors{};
    const auto verify = [&](int variant, const std::vector<float>& actual) {
        errors[variant] = std::max(errors[variant], require_close(actual, reference, 2e-6, 2e-5, names[variant]));
        require_probability_rows(actual, rows, cols);
    };
    const auto initial = Clock::now();
    auto workspace = std::make_unique<SoftmaxWorkspace>();
    auto first = workspace->run(input, rows, cols);
    const double setup_ms = milliseconds(initial);
    verify(2, first);
    setups << rows << ',' << cols << ',' << setup_ms << ',' << workspace->capacity()
           << ',' << workspace->allocation_events() << ',' << seed << '\n';
    const auto call = [&](int variant) {
        if (variant == 0) return softmax_cuda_warp_register(input, rows, cols);
        if (variant == 1) return fresh(input, rows, cols);
        return workspace->run(input, rows, cols);
    };
    for (int i = 0; i < warmup; ++i)
        for (int v = 0; v < 3; ++v) verify(v, call(v));
    std::array<std::vector<double>, 3> timings;
    const auto allocations_before = workspace->allocation_events();
    for (int sample = 0; sample < repeats; ++sample) {
        for (int slot = 0; slot < 3; ++slot) {
            const int variant = (sample + slot) % 3;
            const auto begin = Clock::now();
            auto output = call(variant);
            const double elapsed = milliseconds(begin);
            verify(variant, output); // outside timing; same validation for all variants
            if (!std::isfinite(elapsed) || elapsed <= 0) throw std::runtime_error("invalid timing");
            timings[variant].push_back(elapsed);
            samples << rows << ',' << cols << ',' << names[variant] << ',' << sample << ','
                    << slot << ',' << elapsed << ',' << seed << '\n';
        }
    }
    // Separate diagnostic pass. Not mixed into the uninstrumented performance medians.
    for (int sample = 0; sample < repeats; ++sample) {
        for (int slot = 0; slot < 2; ++slot) {
            const int variant = 1 + (sample + slot) % 2;
            SoftmaxStageTimes t;
            std::vector<float> output;
            const auto begin = Clock::now();
            if (variant == 1) {
                SoftmaxWorkspace temporary;
                output = temporary.run_profiled(input, rows, cols, t);
            } else output = workspace->run_profiled(input, rows, cols, t);
            const double wall = milliseconds(begin);
            verify(variant, output);
            stages << rows << ',' << cols << ',' << names[variant] << ',' << sample << ',' << slot
                   << ',' << t.validation_ms << ',' << t.reserve_ms << ',' << t.h2d_ms
                   << ',' << t.kernel_sync_ms << ',' << t.host_output_ms << ',' << t.d2h_ms
                   << ',' << t.total_ms << ',' << wall - t.total_ms << ',' << wall
                   << ',' << (variant == 1 ? 1 : 0) << ',' << seed << '\n';
        }
    }
    if (workspace->allocation_events() != allocations_before || allocations_before != 1)
        throw std::runtime_error("steady workspace unexpectedly allocated");
    for (int v = 0; v < 3; ++v) {
        const auto& values = timings[v];
        const double ms = median(values);
        summary << rows << ',' << cols << ',' << names[v] << ',' << ms << ','
                << *std::min_element(values.begin(), values.end()) << ',' << *std::max_element(values.begin(), values.end())
                << ',' << median(timings[0])/ms << ',' << median(timings[1])/ms << ',' << errors[v]
                << ",2e-6,2e-5," << warmup << ',' << repeats << ',' << seed << ",PASS\n";
    }
    std::cout << "WORKSPACE_BENCH_CASE_PASS " << rows << 'x' << cols << " reuse_allocations=0\n";
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 3 || std::string(argv[1]) != "--output-dir")
            throw std::invalid_argument("usage: softmax_workspace_benchmark --output-dir EXISTING_EMPTY_DIR");
        device::check(cudaSetDevice(0), "benchmark device");
        device::check(cudaFree(nullptr), "initialize CUDA context"); // excluded from measurements
        const std::filesystem::path directory(argv[2]);
        auto summary = open_csv(directory, "workspace_summary.csv");
        auto samples = open_csv(directory, "workspace_samples.csv");
        auto stages = open_csv(directory, "workspace_stages.csv");
        auto setups = open_csv(directory, "workspace_setup.csv");
        summary << "rows,cols,implementation,median_ms,min_ms,max_ms,speedup_vs_wrapper,speedup_vs_fresh,max_abs_error,atol,rtol,warmup,repeats,seed,correctness\n";
        samples << "rows,cols,implementation,sample,order_slot,elapsed_ms,seed\n";
        stages << "rows,cols,implementation,sample,order_slot,validation_ms,reserve_ms,h2d_ms,kernel_sync_ms,host_output_ms,d2h_ms,total_ms,lifecycle_ms,wall_total_ms,allocation_events,seed\n";
        setups << "rows,cols,first_call_ms,capacity,allocation_events,seed\n";
        const std::pair<std::size_t, std::size_t> shapes[] = {{1,128}, {7,37}, {1024,128}, {1024,257},
            {1024,512}, {1024,1024}, {1024,2048}, {1024,4097}};
        for (const auto& shape : shapes) run_case(shape.first, shape.second, summary, samples, stages, setups);
        for (auto* stream : {&summary, &samples, &stages, &setups}) {
            stream->flush(); if (!*stream) throw std::runtime_error("CSV write failed");
        }
        std::cout << "WORKSPACE_BENCH_PASS summary=24 raw=504 stages=336 setup=8\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL " << error.what() << '\n'; return 1; }
}
