#include <torch/extension.h>

#include <iostream>

torch::Tensor ibf_dense_cuda(torch::Tensor poly);

#define CHECK_SIZE(x) TORCH_CHECK(x.dense_dim() == 3, #x " must be have dim == 3")
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_SIZE(x);

torch::Tensor ibf_dense(
        torch::Tensor poly) {
        CHECK_INPUT(poly);
        return ibf_dense_cuda(poly);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ibf_dense",
 	&ibf_dense,
	"Convert a Bernstein polynomial in implifict form to Explicit form");
}

