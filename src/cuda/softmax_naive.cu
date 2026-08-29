#include "kernel_lab/cuda_kernels.hpp"
#include "kernel_lab/validation.hpp"
#include "device_ops.cuh"
#include <cfloat>

namespace kernel_lab {
namespace {
__global__ void softmax_kernel(const float* input, float* output, const std::size_t cols) {
    extern __shared__ float shared[];
    const std::size_t row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;
    float local_max = -FLT_MAX;
    for (std::size_t col = tid; col < cols; col += blockDim.x) local_max = fmaxf(local_max, row_input[col]);
    shared[tid] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    const float maximum = shared[0];
    // All warps must read the maximum before shared[0] is reused for sums.
    __syncthreads();
    float local_sum = 0;
    for (std::size_t col = tid; col < cols; col += blockDim.x) {
        const float value = expf(row_input[col] - maximum);
        row_output[col] = value;
        local_sum += value;
    }
    shared[tid] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    const float denominator = shared[0];
    for (std::size_t col = tid; col < cols; col += blockDim.x) row_output[col] /= denominator;
}
}  // namespace

void device::launch_softmax(const float* input, float* output, std::size_t rows, std::size_t cols) {
    constexpr unsigned int threads = 256;
    softmax_kernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float)>>>(input, output, cols);
    check(cudaGetLastError(), "launch softmax_kernel");
}
std::vector<float> softmax_cuda_naive(const std::vector<float>& input, std::size_t rows, std::size_t cols) {
    validate_softmax_shape(input.size(), rows, cols);
    require_finite_input(input);
    device::Buffer d_input(input.size()), d_output(input.size());
    d_input.upload(input);
    device::launch_softmax(d_input.data(), d_output.data(), rows, cols);
    device::check(cudaDeviceSynchronize(), "synchronize softmax");
    return d_output.download();
}
}  // namespace kernel_lab
