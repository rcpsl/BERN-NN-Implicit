#include <torch/extension.h>
#include <vector>

// Declare the function that will be implemented in CUDA
torch::Tensor row_convolution_cuda(torch::Tensor T1, torch::Tensor T2);

// Define a check to ensure that the tensors are on CUDA and are contiguous
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This is the C++ interface to the CUDA function
torch::Tensor row_convolution(torch::Tensor T1, torch::Tensor T2) {
    CHECK_INPUT(T1);
    CHECK_INPUT(T2);
    return row_convolution_cuda(T1, T2);
}

// Register the module with PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("row_convolution", &row_convolution, "Performs row-wise convolution between two tensors");
}
