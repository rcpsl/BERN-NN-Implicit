#include <torch/extension.h>

__global__ void repeatTerms2Kernel(float* TA, float* result, int tA, int n, int n_columns, int rows_TA) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = n * (tA * tA + tA) / 2 * n_columns; // Total elements in the output tensor

    if (idx < total_output_elements) {
        int col = idx % n_columns; // Column index
        int output_row = idx / n_columns; // Row index in the output tensor

        // Determine the chunk and the position within the chunk
        int chunk_index = 0, row_within_chunk = 0, accumulated_rows = 0;

        for (int i = 0; i < tA; i++) {
            int rows_in_this_chunk = (tA - i) * n;
            if (output_row < accumulated_rows + rows_in_this_chunk) {
                chunk_index = i;
                row_within_chunk = output_row - accumulated_rows;
                break;
            }
            accumulated_rows += rows_in_this_chunk;
        }

        // Calculate the corresponding row in TA
        int TA_row = chunk_index * n + row_within_chunk;
        if (TA_row < rows_TA) {
            result[idx] = TA[TA_row * n_columns + col];
            // Apply doubling condition
            if ((row_within_chunk >= n) && ((row_within_chunk - n) % n == 0)) {
                result[idx] *= 2;
            }
        } else {
            result[idx] = 0; // If out of bounds, set to zero
        }
    }
}

torch::Tensor repeat_terms_2_cuda(torch::Tensor TA, int tA, int n, int n_columns) {
    int rows_TA = TA.size(0);
    int total_output_rows = n * (tA * tA + tA) / 2;
    auto result = torch::empty({total_output_rows, n_columns}, TA.options());
    int threads = 1024;
    int blocks = (total_output_rows * n_columns + threads - 1) / threads;

    repeatTerms2Kernel<<<blocks, threads>>>(TA.data_ptr<float>(), result.data_ptr<float>(), tA, n, n_columns, rows_TA);

    return result;
}
