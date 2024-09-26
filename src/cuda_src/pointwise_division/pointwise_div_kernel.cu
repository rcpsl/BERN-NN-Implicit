#include <torch/extension.h>

__global__ void pointwiseDivKernel(const float* T1, const float* T2, float* result, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;
        if (T2[index] != 0) {
            result[index] = T1[index] / T2[index];
        } else {
            result[index] = 0;
        }
    }
}

torch::Tensor pointwise_division_cuda(torch::Tensor T1, torch::Tensor T2) {
    auto result = torch::zeros_like(T1);
    int rows = T1.size(0);
    int cols = T1.size(1);

    // dim3 threadsPerBlock(32, 32);
    dim3 block(32, 32);
    // dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, 
    //                (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dim3 grid((cols + block.x - 1) / block.x, 
                   (rows + block.y - 1) / block.y);

    pointwiseDivKernel<<<grid, block>>>(T1.data_ptr<float>(), T2.data_ptr<float>(), result.data_ptr<float>(), rows, cols);

    return result;
}
