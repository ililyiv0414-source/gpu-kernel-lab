#include "kernel_lab/softmax.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kernel_lab {

std::vector<float> softmax_reference(
    const std::vector<float>& input,
    const std::size_t rows,
    const std::size_t cols) {
    if (rows == 0 || cols == 0 || rows > std::numeric_limits<std::size_t>::max() / cols || input.size() != rows * cols) {
        throw std::invalid_argument("softmax input shape is invalid");
    }

    std::vector<float> output(input.size());
    for (std::size_t row = 0; row < rows; ++row) {
        const auto begin = input.begin() + static_cast<std::ptrdiff_t>(row * cols);
        const auto end = begin + static_cast<std::ptrdiff_t>(cols);
        const float maximum = *std::max_element(begin, end);

        double sum = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            const float value = std::exp(input[row * cols + col] - maximum);
            output[row * cols + col] = value;
            sum += value;
        }
        if (!(sum > 0.0) || !std::isfinite(sum)) {
            throw std::runtime_error("softmax normalization is not finite");
        }
        for (std::size_t col = 0; col < cols; ++col) {
            output[row * cols + col] = static_cast<float>(output[row * cols + col] / sum);
        }
    }
    return output;
}

float max_abs_error(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("vectors must have the same length");
    }
    float error = 0.0F;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (!std::isfinite(lhs[i]) || !std::isfinite(rhs[i])) {
            throw std::invalid_argument("error comparison requires finite values");
        }
        error = std::max(error, std::abs(lhs[i] - rhs[i]));
    }
    return error;
}

}  // namespace kernel_lab
