#include <torch/extension.h>

torch::Tensor find_indices_cuda(torch::Tensor degree, int64_t batch_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("find_indices", &find_indices_cuda, "Find indices CUDA implementation");
}
