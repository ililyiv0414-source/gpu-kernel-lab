#pragma once

#include <cstddef>
#include <vector>

namespace kernel_lab {

// These functions require a CUDA build. Performance claims are invalid until
// they are compiled and measured on a named GPU with saved raw results.
// Only finite FP32 inputs are supported. Wrappers validate inputs, allocate
// device buffers, transfer H2D/D2H and synchronize before returning.
std::vector<float> softmax_cuda_naive(
    const std::vector<float>& input,
    std::size_t rows,
    std::size_t cols);

// Same finite-FP32 contract as the baseline. The warp-register variant uses
// a block-shuffle fallback for rows wider than 1024 elements.
std::vector<float> softmax_cuda_block_shuffle(
    const std::vector<float>& input, std::size_t rows, std::size_t cols);
std::vector<float> softmax_cuda_warp_register(
    const std::vector<float>& input, std::size_t rows, std::size_t cols);

std::vector<float> convolution_cuda_naive(
    const std::vector<float>& signal,
    const std::vector<float>& kernel);

}  // namespace kernel_lab
