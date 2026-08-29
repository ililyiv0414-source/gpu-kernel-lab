#include "kernel_lab/cuda_kernels.hpp"
#include "operator_cases.hpp"
#include "../src/cuda/device_ops.cuh"
#include <limits>

int main() {
    try {
        kernel_lab::device::check(cudaSetDevice(0), "select CUDA GPU");
        const cases::Softmax variants[] = {kernel_lab::softmax_cuda_block_shuffle,
                                           kernel_lab::softmax_cuda_warp_register};
        const char* names[] = {"block_shuffle", "warp_register"};
        for (int variant = 0; variant < 2; ++variant) {
            const auto& softmax = variants[variant];
            const int common = cases::run(softmax, kernel_lab::convolution_cuda_naive);
            int boundaries = 0;
            for (std::size_t rows : {1U, 3U, 4U, 5U, 7U}) {
                for (std::size_t cols : {31U, 32U, 33U, 127U, 128U, 129U, 511U, 512U,
                                         513U, 1023U, 1024U, 1025U, 2048U, 4097U}) {
                    std::vector<float> input(rows * cols);
                    for (std::size_t i = 0; i < input.size(); ++i)
                        input[i] = float(int(i % 29) - 14) * 0.25f;
                    const auto actual = softmax(input, rows, cols);
                    kernel_lab::require_close(actual, cases::softmax_oracle(input, rows, cols),
                                               2e-6, 2e-5, names[variant]);
                    kernel_lab::require_probability_rows(actual, rows, cols);
                    ++boundaries;
                }
            }
            cases::rejects<std::invalid_argument>([&] { softmax({}, 0, 1); });
            cases::rejects<std::invalid_argument>([&] { softmax({1}, 1, 2); });
            cases::rejects<std::invalid_argument>([&] { softmax({}, std::numeric_limits<std::size_t>::max(), 2); });
            cases::rejects<std::invalid_argument>([&] { softmax({std::numeric_limits<float>::quiet_NaN()}, 1, 1); });
            cases::rejects<std::invalid_argument>([&] { softmax({std::numeric_limits<float>::infinity()}, 1, 1); });
            std::cout << "OPTIMIZED_PASS " << names[variant] << " common=" << common
                      << " boundary=" << boundaries << " invalid=5\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
