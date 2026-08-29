#include "kernel_lab/convolution.hpp"
#include "kernel_lab/softmax.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

template <typename Function>
double median_ms(Function&& function, const int warmup, const int repeats) {
    for (int i = 0; i < warmup; ++i) {
        function();
    }
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        const auto start = std::chrono::steady_clock::now();
        function();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const std::string output_path = argc > 1 ? argv[1] : "cpu_benchmark.csv";
    std::mt19937 generator(20260825U);
    std::uniform_real_distribution<float> distribution(-3.0F, 3.0F);

    std::ofstream csv(output_path);
    if (!csv) {
        std::cerr << "cannot open output file: " << output_path << '\n';
        return 1;
    }
    csv << "operator,implementation,rows,cols_or_kernel,median_ms,repeats,seed\n";

    volatile float sink = 0.0F;
    for (const std::size_t cols : {128U, 512U, 1024U}) {
        const std::size_t rows = 1024;
        std::vector<float> input(rows * cols);
        for (float& value : input) value = distribution(generator);
        const double elapsed = median_ms([&] {
            const auto output = kernel_lab::softmax_reference(input, rows, cols);
            sink += output.front();
        }, 2, 9);
        csv << "softmax,cpu_reference," << rows << ',' << cols << ',' << std::fixed
            << std::setprecision(4) << elapsed << ",9,20260825\n";
    }

    for (const std::size_t kernel_size : {15U, 63U, 255U}) {
        std::vector<float> signal(1U << 15U);
        std::vector<float> kernel(kernel_size);
        for (float& value : signal) value = distribution(generator);
        for (float& value : kernel) value = distribution(generator);
        const double elapsed = median_ms([&] {
            const auto output = kernel_lab::convolution_blocked(signal, kernel, 1024);
            sink += output.front();
        }, 1, 5);
        csv << "convolution,cpu_blocked," << signal.size() << ',' << kernel_size << ','
            << std::fixed << std::setprecision(4) << elapsed << ",5,20260825\n";
    }

    std::cout << "wrote " << output_path << " (sink=" << sink << ")\n";
    return 0;
}
