#include <torch/library.h>

TORCH_LIBRARY(kernel_lab_ext, m) {
    m.def("softmax(Tensor input) -> Tensor");
}
