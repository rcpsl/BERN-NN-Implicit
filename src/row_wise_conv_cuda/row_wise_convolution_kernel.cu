#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for row-wise convolution
__global__ void row_convolution_kernel(const float *T1, const float *T2, float *result, int T1_rows, int T1_cols, int T2_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int result_cols = T1_cols + T2_cols - 1;

    if (row < T1_rows && col < result_cols) {
        float sum = 0.0f;
        // Calculate the starting and ending indices for T2
        int start_k = (col >= T2_cols) ? col - T2_cols + 1 : 0;
        int end_k = min(col, T1_cols - 1);

        // Perform the convolution
        for (int k = start_k; k <= end_k; ++k) {
            sum += T1[row * T1_cols + k] * T2[row * T2_cols + col - k];
        }
        result[row * result_cols + col] = sum;
    }
}

// Host function that PyTorch calls
torch::Tensor row_convolution_cuda(torch::Tensor T1, torch::Tensor T2) {
    const auto T1_rows = T1.size(0);
    const auto T1_cols = T1.size(1);
    const auto T2_cols = T2.size(1);
    const auto result_cols = T1_cols + T2_cols - 1;

    // Allocate result tensor
    auto result = torch::zeros({T1_rows, result_cols}, T1.options());

    // Calculate grid and block sizes
    // const dim3 threads(32, 32);
    dim3 block(32, 32);

    // TODO: need to be careful that we don't allocate too many blocks.
    // if T1_rows + block.y - 1) / block.y is larger than max number
    // of allowed block (2^16), then there may be an error

    dim3 grid((result_cols + block.x - 1) / block.x, (T1_rows + block.y - 1) / block.y);

    // Launch kernel
    row_convolution_kernel<<<grid, block>>>(
        T1.data_ptr<float>(), T2.data_ptr<float>(), result.data_ptr<float>(), 
        T1_rows, T1_cols, T2_cols
    );

    // Wait for the GPU to finish before returning to Python
    cudaDeviceSynchronize();

    return result;
}
