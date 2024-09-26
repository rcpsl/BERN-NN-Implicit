#include <torch/extension.h>

torch::Tensor pointwise_division_cuda(torch::Tensor T1, torch::Tensor T2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pointwise_division", &pointwise_division_cuda, "Pointwise division CUDA implementation");
}
