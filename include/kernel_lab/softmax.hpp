#pragma once

#include <cstddef>
#include <vector>

namespace kernel_lab {

std::vector<float> softmax_reference(
    const std::vector<float>& input,
    std::size_t rows,
    std::size_t cols);

float max_abs_error(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs);

}  // namespace kernel_lab
