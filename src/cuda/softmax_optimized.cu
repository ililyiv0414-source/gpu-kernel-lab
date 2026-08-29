#include "kernel_lab/cuda_kernels.hpp"
#include "kernel_lab/validation.hpp"
#include "device_ops.cuh"
#include <cfloat>

namespace kernel_lab {
namespace {
constexpr unsigned full_warp = 0xffffffffU;
template<bool Maximum>
__device__ __forceinline__ float combine(float a, float b) {
    return Maximum ? fmaxf(a, b) : a + b;
}
template<bool Maximum>
__device__ __forceinline__ float warp_reduce(float value) {
    // Every lane of a participating warp executes each shuffle with this mask.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = combine<Maximum>(value, __shfl_down_sync(full_warp, value, offset));
    return __shfl_sync(full_warp, value, 0);
}
template<bool Maximum>
__device__ float block_reduce(float value, float* partial) {
    const unsigned lane = threadIdx.x & 31U, warp = threadIdx.x >> 5U;
    value = warp_reduce<Maximum>(value);
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < 8 ? partial[lane] : (Maximum ? -FLT_MAX : 0.0f);
        value = warp_reduce<Maximum>(value);
        if (lane == 0) partial[0] = value;
    }
    __syncthreads();
    const float result = partial[0];
    // Every warp must finish reading before the next reduction reuses partial.
    __syncthreads();
    return result;
}
__global__ void softmax_block_shuffle_kernel(const float* input, float* output, std::size_t cols) {
    __shared__ float partial[8]; // launch is exactly 256 threads / 8 full warps
    const auto row_offset = std::size_t(blockIdx.x) * cols;
    float maximum = -FLT_MAX;
    for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x)
        maximum = fmaxf(maximum, input[row_offset + col]);
    maximum = block_reduce<true>(maximum, partial);
    float denominator = 0;
    for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = expf(input[row_offset + col] - maximum);
        output[row_offset + col] = value;
        denominator += value;
    }
    denominator = block_reduce<false>(denominator, partial);
    for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x)
        output[row_offset + col] /= denominator;
}
template<int Items>
__global__ void softmax_warp_register_kernel(const float* input, float* output,
                                            std::size_t rows, std::size_t cols) {
    const unsigned lane = threadIdx.x & 31U;
    const std::size_t row = std::size_t(blockIdx.x) * 4 + (threadIdx.x >> 5U);
    // Uniform across a whole warp; valid rows keep all 32 lanes participating.
    if (row >= rows) return;
    float values[Items];
    float maximum = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < Items; ++i) {
        const auto col = std::size_t(lane) + std::size_t(i) * 32;
        values[i] = col < cols ? input[row * cols + col] : -FLT_MAX;
        maximum = fmaxf(maximum, values[i]);
    }
    maximum = warp_reduce<true>(maximum);
    float denominator = 0;
    #pragma unroll
    for (int i = 0; i < Items; ++i) {
        const auto col = std::size_t(lane) + std::size_t(i) * 32;
        values[i] = col < cols ? expf(values[i] - maximum) : 0.0f;
        denominator += values[i];
    }
    denominator = warp_reduce<false>(denominator);
    #pragma unroll
    for (int i = 0; i < Items; ++i) {
        const auto col = std::size_t(lane) + std::size_t(i) * 32;
        if (col < cols) output[row * cols + col] = values[i] / denominator;
    }
}
template<class Launch>
std::vector<float> wrapped(const std::vector<float>& input, std::size_t rows,
                           std::size_t cols, Launch launch) {
    validate_softmax_shape(input.size(), rows, cols);
    require_finite_input(input);
    device::Buffer d_input(input.size()), d_output(input.size());
    d_input.upload(input);
    launch(d_input.data(), d_output.data(), rows, cols);
    device::check(cudaDeviceSynchronize(), "synchronize optimized softmax");
    return d_output.download();
}
} // namespace

void device::launch_softmax_block_shuffle(const float* input, float* output,
                                         std::size_t rows, std::size_t cols) {
    launch_softmax_block_shuffle_on_stream(input, output, rows, cols, nullptr);
}
void device::launch_softmax_block_shuffle_on_stream(const float* input, float* output,
                                                    std::size_t rows, std::size_t cols,
                                                    cudaStream_t stream) {
    softmax_block_shuffle_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(input, output, cols);
    check(cudaGetLastError(), "launch softmax_block_shuffle_kernel");
}
void device::launch_softmax_warp_register(const float* input, float* output,
                                         std::size_t rows, std::size_t cols) {
    launch_softmax_warp_register_on_stream(input, output, rows, cols, nullptr);
}
void device::launch_softmax_warp_register_on_stream(const float* input, float* output,
                                                    std::size_t rows, std::size_t cols,
                                                    cudaStream_t stream) {
    const auto blocks = static_cast<unsigned>((rows + 3) / 4);
    if (cols <= 32) softmax_warp_register_kernel<1><<<blocks, 128, 0, stream>>>(input, output, rows, cols);
    else if (cols <= 128) softmax_warp_register_kernel<4><<<blocks, 128, 0, stream>>>(input, output, rows, cols);
    else if (cols <= 512) softmax_warp_register_kernel<16><<<blocks, 128, 0, stream>>>(input, output, rows, cols);
    else if (cols <= 1024) softmax_warp_register_kernel<32><<<blocks, 128, 0, stream>>>(input, output, rows, cols);
    else { launch_softmax_block_shuffle_on_stream(input, output, rows, cols, stream); return; }
    check(cudaGetLastError(), "launch softmax_warp_register_kernel");
}
std::vector<float> softmax_cuda_block_shuffle(const std::vector<float>& input,
                                             std::size_t rows, std::size_t cols) {
    return wrapped(input, rows, cols, device::launch_softmax_block_shuffle);
}
std::vector<float> softmax_cuda_warp_register(const std::vector<float>& input,
                                             std::size_t rows, std::size_t cols) {
    return wrapped(input, rows, cols, device::launch_softmax_warp_register);
}
} // namespace kernel_lab
