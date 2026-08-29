#include "kernel_lab/cuda_kernels.hpp"
#include "kernel_lab/validation.hpp"
#include "device_ops.cuh"

namespace kernel_lab {
namespace {
__global__ void convolution_kernel(const float* signal, int n, const float* kernel,
                                   int k, float* output, int output_size) {
    const std::size_t wide_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (wide_index >= static_cast<std::size_t>(output_size)) return;
    const int index = static_cast<int>(wide_index);
    const int begin = index - k + 1 > 0 ? index - k + 1 : 0;
    const int end = index < n - 1 ? index : n - 1;
    float value = 0;
    for (int i = begin; i <= end; ++i) value += signal[i] * kernel[index - i];
    output[index] = value;
}
}  // namespace

void device::launch_convolution(const float* signal, const float* kernel, float* output,
                               std::size_t n, std::size_t k) {
    const auto size = checked_convolution_size(n, k);
    constexpr unsigned int threads = 256;
    const auto blocks = static_cast<unsigned int>((size - 1) / threads + 1);
    convolution_kernel<<<blocks, threads>>>(signal, static_cast<int>(n), kernel,
                                            static_cast<int>(k), output, static_cast<int>(size));
    check(cudaGetLastError(), "launch convolution_kernel");
}
std::vector<float> convolution_cuda_naive(const std::vector<float>& signal, const std::vector<float>& kernel) {
    const auto size = checked_convolution_size(signal.size(), kernel.size());
    require_finite_input(signal);
    require_finite_input(kernel);
    device::Buffer d_signal(signal.size()), d_kernel(kernel.size()), d_output(size);
    d_signal.upload(signal);
    d_kernel.upload(kernel);
    device::launch_convolution(d_signal.data(), d_kernel.data(), d_output.data(), signal.size(), kernel.size());
    device::check(cudaDeviceSynchronize(), "synchronize convolution");
    return d_output.download();
}
}  // namespace kernel_lab
