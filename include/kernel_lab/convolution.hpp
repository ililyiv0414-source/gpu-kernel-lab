#pragma once

#include <cstddef>
#include <vector>

namespace kernel_lab {

std::vector<float> convolution_direct(
    const std::vector<float>& signal,
    const std::vector<float>& kernel);

std::vector<float> convolution_blocked(
    const std::vector<float>& signal,
    const std::vector<float>& kernel,
    std::size_t block_size);

}  // namespace kernel_lab
