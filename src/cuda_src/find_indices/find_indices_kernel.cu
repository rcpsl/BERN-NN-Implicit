#include <torch/extension.h>

__host__ __device__
int64_t int_pow(int base, int exp) {
    int64_t accum = 1;
    for (int i = 0; i < exp; ++i) {
        accum *= base;
    }
    return accum;
}

__global__ void findIndicesBatchKernel(const int* degree, int n, int64_t* output, int64_t batch_start, int64_t batch_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx += batch_start; // Adjust idx for the current batch
    if (idx >= batch_start + batch_size) return;

    int64_t quotient = idx;
    for (int dim = n - 1; dim >= 0; --dim) {
        int current_degree = degree[dim];
        output[(idx - batch_start) * n + dim] = quotient % (current_degree + 1);
        quotient /= (current_degree + 1);
    }
}

torch::Tensor find_indices_cuda(torch::Tensor degree, int64_t batch_size) {
    const int n = degree.size(0);
    int64_t max_elements = 1;
    for (int i = 0; i < n; ++i) {
        max_elements *= static_cast<int64_t>(degree[i].item<int>() + 1);
    }

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    int64_t num_batches = (max_elements + batch_size - 1) / batch_size;
    torch::Tensor all_output = torch::empty({0, n}, options);

    for (int64_t batch = 0; batch < num_batches; ++batch) {
        int64_t batch_start = batch * batch_size;
        int64_t current_batch_size = std::min(batch_size, max_elements - batch_start);

        torch::Tensor output = torch::empty({current_batch_size, n}, options);
        int64_t threads = 1024;
        int64_t blocks = std::min((current_batch_size + threads - 1) / threads, int_pow(2, 30));

        findIndicesBatchKernel<<<blocks, threads>>>(degree.data_ptr<int>(), n, output.data_ptr<int64_t>(), batch_start, current_batch_size);

        // Append the output of this batch to all_output
        all_output = torch::cat({all_output, output}, 0);
    }

    return all_output;
}
