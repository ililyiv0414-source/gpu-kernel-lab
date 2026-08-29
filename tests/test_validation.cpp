#include "kernel_lab/convolution.hpp"
#include "kernel_lab/softmax.hpp"
#include "operator_cases.hpp"
#include <limits>

int main() {
    try {
        const int count = cases::run(kernel_lab::softmax_reference, kernel_lab::convolution_direct);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const float inf = std::numeric_limits<float>::infinity();
        cases::rejects([&] { kernel_lab::require_close({nan}, {0}, 1, 1, "nan"); });
        cases::rejects([&] { kernel_lab::require_close({inf}, {inf}, 1, 1, "inf"); });
        cases::rejects([] { kernel_lab::require_close({2}, {0}, 1e-6, 1e-6, "wrong"); });
        cases::rejects([] { kernel_lab::require_close({1, 2}, {1}, 0, 0, "size"); });
        cases::rejects([] { kernel_lab::require_close({}, {}, 0, 0, "empty"); });
        cases::rejects([] { kernel_lab::require_probability_rows({0.2F, 0.2F}, 1, 2); });
        cases::rejects([&] { kernel_lab::max_abs_error({nan}, {0}); });
        cases::rejects([] { kernel_lab::validate_softmax_shape(1, 0, 1); });
        cases::rejects([] { kernel_lab::validate_softmax_shape(1, 1, 2); });
        cases::rejects([] { kernel_lab::validate_softmax_shape(0, std::numeric_limits<std::size_t>::max(), 2); });
        cases::rejects([] { kernel_lab::checked_convolution_size(0, 1); });
        cases::rejects([] { kernel_lab::checked_convolution_size(INT_MAX, 2); });
        cases::rejects([&] { kernel_lab::require_finite_input({nan}); });
        cases::rejects([&] { kernel_lab::require_finite_input({inf}); });
        cases::rejects([] { kernel_lab::softmax_reference({}, std::numeric_limits<std::size_t>::max(), 2); });
        kernel_lab::require_close({1.000001F}, {1}, 2e-6, 0, "within_tolerance");
        if (kernel_lab::checked_convolution_size(3, 2) != 4) throw std::runtime_error("convolution shape");
        std::cout << "PASS " << count << " CPU oracle cases and 17 validation checks\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
