#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void row_convolution_kernel(const float *T1, const float *T2, float *result, int T1_rows, int T1_cols, int T2_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int start_col = blockIdx.x * blockDim.x + threadIdx.x;
    int result_cols = T1_cols + T2_cols - 1;

    while (row < T1_rows) {
      int col = start_col;
      while (col < result_cols) {
        float sum = 0.0f;
        int start_k = (col >= T2_cols) ? col - T2_cols + 1 : 0;
        int end_k = min(col, T1_cols - 1);

        // Perform the convolution
        for (int k = start_k; k <= end_k; ++k) {
            sum += T1[row * T1_cols + k] * T2[row * T2_cols + col - k];
        }
        result[row * result_cols + col] = sum;
	col += gridDim.x;
      }
      row += gridDim.y;
    }
}

torch::Tensor row_convolution_cuda(torch::Tensor T1, torch::Tensor T2) {
    const auto T1_rows = T1.size(0);
    const auto T1_cols = T1.size(1);
    const auto T2_cols = T2.size(1);
    const auto result_cols = T1_cols + T2_cols - 1;
    auto result = torch::zeros({T1_rows, result_cols}, T1.options());

    dim3 block(32, 32);

    int target_grid_dim_x = (result_cols + block.x - 1) / block.x;
    int target_grid_dim_y = (T1_rows + block.y - 1) / block.y;

    // Cap the grid dimensions, as this kernel can easily allocate too many blocks.
    int grid_dim_x = std::min(target_grid_dim_x, static_cast<int>(std::pow(2, 16) - 1));
    int grid_dim_y = std::min(target_grid_dim_y, static_cast<int>(std::pow(2, 16) - 1));
    dim3 grid(grid_dim_x, grid_dim_y);

    row_convolution_kernel<<<grid, block>>>(
        T1.data_ptr<float>(), T2.data_ptr<float>(), result.data_ptr<float>(), 
        T1_rows, T1_cols, T2_cols
    );

    cudaDeviceSynchronize();

    return result;
}
