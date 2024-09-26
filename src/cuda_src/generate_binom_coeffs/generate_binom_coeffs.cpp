#include <torch/extension.h>
torch::Tensor generate_binom_coeffs_cuda(torch::Tensor L);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("generate_binom_coeffs", &generate_binom_coeffs_cuda, "Generate binomial coefficients CUDA implementation");
}
