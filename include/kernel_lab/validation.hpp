#pragma once
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace kernel_lab {
inline void validate_softmax_shape(std::size_t count, std::size_t rows, std::size_t cols) {
    if (!rows || !cols || rows > static_cast<std::size_t>(INT_MAX) ||
        cols > static_cast<std::size_t>(INT_MAX) ||
        rows > std::numeric_limits<std::size_t>::max() / cols || count != rows * cols) {
        throw std::invalid_argument("invalid softmax shape or integer range");
    }
}
inline std::size_t checked_convolution_size(std::size_t n, std::size_t k) {
    const auto limit = static_cast<std::size_t>(INT_MAX);
    if (!n || !k || n > limit || k > limit || n > limit - k + 1) {
        throw std::invalid_argument("invalid convolution size or integer range");
    }
    return n + k - 1;
}
inline void require_finite_input(const std::vector<float>& values) {
    for (float value : values) {
        if (!std::isfinite(value)) throw std::invalid_argument("only finite inputs are supported");
    }
}
// Elementwise: |actual-reference| <= atol + rtol*|reference|.
// Explicitly reject NaN/Inf, which max(error, NaN) could otherwise hide.
inline double require_close(const std::vector<float>& actual,
                            const std::vector<float>& reference,
                            double atol, double rtol, const std::string& label) {
    if (actual.size() != reference.size() || actual.empty() ||
        !std::isfinite(atol) || !std::isfinite(rtol) || atol < 0 || rtol < 0) {
        throw std::runtime_error(label + ": invalid comparison arguments");
    }
    double maximum = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], b = reference[i];
        const double error = std::abs(a - b);
        if (!std::isfinite(a) || !std::isfinite(b) || error > atol + rtol * std::abs(b)) {
            throw std::runtime_error(label + ": mismatch at index " + std::to_string(i));
        }
        maximum = std::max(maximum, error);
    }
    return maximum;
}
inline void require_probability_rows(const std::vector<float>& values,
                                     std::size_t rows, std::size_t cols) {
    validate_softmax_shape(values.size(), rows, cols);
    for (std::size_t row = 0; row < rows; ++row) {
        double sum = 0;
        for (std::size_t col = 0; col < cols; ++col) {
            const float value = values[row * cols + col];
            if (!std::isfinite(value) || value < 0 || value > 1) {
                throw std::runtime_error("invalid softmax probability");
            }
            sum += value;
        }
        if (std::abs(sum - 1.0) > 2e-5) throw std::runtime_error("softmax row sum mismatch");
    }
}
}  // namespace kernel_lab
