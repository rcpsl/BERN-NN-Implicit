#include <torch/extension.h>
torch::Tensor repeat_terms_cuda(torch::Tensor TA, int tA, int n, int n_columns);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("repeat_terms", &repeat_terms_cuda, "Repeat terms CUDA implementation");
}
