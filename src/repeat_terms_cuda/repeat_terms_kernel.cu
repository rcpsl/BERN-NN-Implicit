#include <torch/extension.h>

__global__ void repeatTermsKernel(const float *TA, float *result, int tA, int n, int n_columns) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rows_per_chunk = n;
    int total_output_rows = n * (tA * (tA + 1) / 2);
    int total_output_elements = total_output_rows * n_columns;

    if (idx < total_output_elements) {
        int col = idx % n_columns; // Column index
        int output_row = idx / n_columns; // Global row index in the output tensor

        // Determine the chunk and the position within the chunk
        int chunk = 0, row_within_chunk = 0, accumulated_rows = 0;

        for (int i = 0; i < tA; i++) {
            int rows_in_this_chunk = (tA - i) * rows_per_chunk;
            if (output_row < accumulated_rows + rows_in_this_chunk) {
                chunk = i;
                row_within_chunk = (output_row - accumulated_rows) % rows_per_chunk;
                break;
            }
            accumulated_rows += rows_in_this_chunk;
        }

        // Calculate the corresponding row in TA
        int TA_row = (chunk * rows_per_chunk + row_within_chunk) % (tA * rows_per_chunk);
        result[idx] = TA[TA_row * n_columns + col];
    }
}

torch::Tensor repeat_terms_cuda(torch::Tensor TA, int tA, int n, int n_columns) {
    int total_output_rows = n * (tA * (tA + 1) / 2);
    auto result = torch::empty({total_output_rows, n_columns}, TA.options());
    int threads = 1024;
    int blocks = (total_output_rows * n_columns + threads - 1) / threads;

    repeatTermsKernel<<<blocks, threads>>>(TA.data_ptr<float>(), result.data_ptr<float>(), tA, n, n_columns);

    return result;
}
