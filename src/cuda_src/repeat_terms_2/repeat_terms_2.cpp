#include <torch/extension.h>
torch::Tensor repeat_terms_2_cuda(torch::Tensor TA, int tA, int n, int n_columns);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("repeat_terms_2", &repeat_terms_2_cuda, "Repeat terms 2 CUDA implementation");
}
