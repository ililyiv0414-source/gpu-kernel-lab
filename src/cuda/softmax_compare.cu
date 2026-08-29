#include "kernel_lab/cuda_kernels.hpp"
#include "kernel_lab/softmax.hpp"
#include "kernel_lab/validation.hpp"
#include "device_ops.cuh"
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <random>
#include <utility>

namespace {
using namespace kernel_lab;
constexpr unsigned seed = 20260829U;
using Launch = void (*)(const float*, float*, std::size_t, std::size_t);
using Wrapper = std::vector<float> (*)(const std::vector<float>&, std::size_t, std::size_t);
const char* names[] = {"cuda_naive", "cuda_block_shuffle", "cuda_warp_register"};
const Launch launches[] = {device::launch_softmax, device::launch_softmax_block_shuffle,
                           device::launch_softmax_warp_register};
const Wrapper wrappers[] = {softmax_cuda_naive, softmax_cuda_block_shuffle, softmax_cuda_warp_register};
volatile float sink = 0;
struct Options { std::filesystem::path output = "."; int warmup = 5, repeats = 21, launches = 100; };
Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (++i >= argc) throw std::invalid_argument("missing option value");
        const std::string value = argv[i];
        if (key == "--output-dir") options.output = value;
        else {
            std::size_t used = 0;
            const int n = std::stoi(value, &used);
            if (used != value.size() || n < 1 || n > 1000) throw std::invalid_argument("invalid count (1..1000)");
            if (key == "--warmup") options.warmup = n;
            else if (key == "--repeats") options.repeats = n;
            else if (key == "--launches") options.launches = n;
            else throw std::invalid_argument("unknown option: " + key);
        }
    }
    return options;
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto n = values.size();
    return n % 2 ? values[n/2] : (values[n/2-1] + values[n/2]) / 2;
}
const char* dispatch(int variant, std::size_t cols) {
    if (variant == 0) return "block_shared256";
    if (variant == 1) return "block_shuffle256";
    if (cols <= 32) return "warp_reg1";
    if (cols <= 128) return "warp_reg4";
    if (cols <= 512) return "warp_reg16";
    if (cols <= 1024) return "warp_reg32";
    return "block_shuffle256_fallback";
}
void run_case(std::size_t rows, std::size_t cols, const Options& options,
              std::ofstream& raw, std::ofstream& summary) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-2, 2);
    std::vector<float> input(rows * cols);
    for (auto& value : input) value = distribution(generator);
    const auto reference = softmax_reference(input, rows, cols);
    device::Buffer d_input(input.size()), d_output(input.size());
    d_input.upload(input);
    std::array<double, 3> errors{};
    const auto verify = [&](int variant, const std::vector<float>& actual) {
        errors[variant] = std::max(errors[variant], require_close(actual, reference, 2e-6, 2e-5, names[variant]));
        require_probability_rows(actual, rows, cols);
    };
    // Context setup, correctness and warmup are outside measurements.
    for (int variant = 0; variant < 3; ++variant) {
        verify(variant, wrappers[variant](input, rows, cols));
        launches[variant](d_input.data(), d_output.data(), rows, cols);
        device::check(cudaDeviceSynchronize(), "pre-timing correctness");
        verify(variant, d_output.download());
        for (int i = 0; i < options.warmup; ++i) {
            launches[variant](d_input.data(), d_output.data(), rows, cols);
            const auto output = wrappers[variant](input, rows, cols);
            sink += output.front();
        }
    }
    device::check(cudaDeviceSynchronize(), "compare warmup");
    std::array<std::array<std::vector<double>, 3>, 2> samples;
    const char* scopes[] = {"batched_event_per_launch", "end_to_end_wall"};
    device::Event start, stop;
    for (int sample = 0; sample < options.repeats; ++sample) {
        for (int scope = 0; scope < 2; ++scope) {
            for (int slot = 0; slot < 3; ++slot) {
                const int variant = (sample + slot) % 3;
                double elapsed = 0;
                if (scope == 0) {
                    device::check(cudaEventRecord(start.get()), "compare start");
                    for (int i = 0; i < options.launches; ++i)
                        launches[variant](d_input.data(), d_output.data(), rows, cols);
                    device::check(cudaEventRecord(stop.get()), "compare stop");
                    device::check(cudaEventSynchronize(stop.get()), "compare synchronize");
                    float total = 0;
                    device::check(cudaEventElapsedTime(&total, start.get(), stop.get()), "compare elapsed");
                    elapsed = double(total) / options.launches;
                } else {
                    const auto begin = std::chrono::steady_clock::now();
                    const auto output = wrappers[variant](input, rows, cols);
                    sink += output.front();
                    elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
                }
                if (!std::isfinite(elapsed) || elapsed <= 0) throw std::runtime_error("invalid elapsed time");
                samples[scope][variant].push_back(elapsed);
                raw << rows << ',' << cols << ',' << names[variant] << ',' << dispatch(variant, cols)
                    << ',' << scopes[scope] << ',' << sample << ',' << slot << ','
                    << (scope == 0 ? options.launches : 1) << ',' << elapsed << ',' << seed << '\n';
            }
        }
    }
    for (int variant = 0; variant < 3; ++variant) {
        launches[variant](d_input.data(), d_output.data(), rows, cols);
        device::check(cudaDeviceSynchronize(), "post-timing correctness");
        verify(variant, d_output.download());
        verify(variant, wrappers[variant](input, rows, cols));
    }
    for (int scope = 0; scope < 2; ++scope) {
        for (int variant = 0; variant < 3; ++variant) {
            const auto& values = samples[scope][variant];
            const double ms = median(values);
            summary << rows << ',' << cols << ',' << names[variant] << ',' << dispatch(variant, cols)
                    << ',' << scopes[scope] << ',' << ms << ',' << *std::min_element(values.begin(), values.end())
                    << ',' << *std::max_element(values.begin(), values.end()) << ',' << median(samples[scope][0])/ms
                    << ',' << errors[variant] << ",2e-6,2e-5," << options.warmup << ',' << options.repeats
                    << ',' << (scope == 0 ? options.launches : 1) << ',' << seed << ",PASS\n";
        }
    }
    std::cout << "COMPARE_CASE_PASS " << rows << 'x' << cols << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        device::check(cudaSetDevice(0), "select CUDA GPU");
        const auto summary_path = options.output / "softmax_compare_summary.csv";
        const auto raw_path = options.output / "softmax_compare_samples.csv";
        if (std::filesystem::exists(summary_path) || std::filesystem::exists(raw_path))
            throw std::runtime_error("refusing to overwrite comparison results");
        std::ofstream summary(summary_path), raw(raw_path);
        if (!summary || !raw) throw std::runtime_error("output directory missing or unwritable");
        for (auto* stream : {&summary, &raw}) {
            stream->imbue(std::locale::classic());
            *stream << std::setprecision(12);
        }
        summary << "rows,cols,implementation,dispatch,scope,median_ms,min_ms,max_ms,baseline_speedup,max_abs_error,atol,rtol,warmup,repeats,launches,seed,correctness\n";
        raw << "rows,cols,implementation,dispatch,scope,sample,order_slot,launches,elapsed_ms,seed\n";
        const std::pair<std::size_t, std::size_t> shapes[] = {
            {1, 128}, {7, 37}, {1024, 128}, {1024, 257}, {1024, 512},
            {1024, 1024}, {1024, 2048}, {1024, 4097}
        };
        for (const auto& shape : shapes) run_case(shape.first, shape.second, options, raw, summary);
        summary.flush(); raw.flush();
        if (!summary || !raw) throw std::runtime_error("comparison CSV write failed");
        std::cout << "SOFTMAX_COMPARE_PASS 48 summary rows; sink=" << sink << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
