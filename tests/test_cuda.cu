#include "kernel_lab/cuda_kernels.hpp"
#include "operator_cases.hpp"
#include "../src/cuda/device_ops.cuh"
#include <limits>

int main() {
    try {
        int devices = 0;
        kernel_lab::device::check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        if (!devices) throw std::runtime_error("no CUDA GPU; this is a failure, not a skipped PASS");
        kernel_lab::device::check(cudaSetDevice(0), "cudaSetDevice");
        const int count = cases::run(kernel_lab::softmax_cuda_naive, kernel_lab::convolution_cuda_naive);
        cases::rejects<std::invalid_argument>([] { kernel_lab::softmax_cuda_naive({}, 0, 1); });
        cases::rejects<std::invalid_argument>([] { kernel_lab::softmax_cuda_naive({1}, 1, 2); });
        cases::rejects<std::invalid_argument>([] { kernel_lab::softmax_cuda_naive({}, std::numeric_limits<std::size_t>::max(), 2); });
        cases::rejects<std::invalid_argument>([] { kernel_lab::convolution_cuda_naive({}, {1}); });
        cases::rejects<std::invalid_argument>([] { kernel_lab::convolution_cuda_naive({1}, {}); });
        cases::rejects<std::invalid_argument>([] { kernel_lab::softmax_cuda_naive({std::numeric_limits<float>::quiet_NaN()}, 1, 1); });
        cases::rejects<std::invalid_argument>([] { kernel_lab::convolution_cuda_naive({1}, {std::numeric_limits<float>::infinity()}); });
        std::cout << "PASS " << count << " GPU oracle cases and 7 invalid-input checks\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
