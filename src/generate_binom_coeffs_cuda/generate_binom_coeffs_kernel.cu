#include <torch/extension.h>
#include <math.h>

__device__ float binom_coeff(int n, int k) {
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;

    float result = 1;
    for (int i = 1; i <= k; ++i) {
        result *= (n - (k - i));
        result /= i;
    }
    return result;
}

__global__ void generateBinomCoeffsKernel(const float* L, float* result, int max_L, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * max_L) {
        int n = L[idx / max_L];
        int k = idx % max_L;

        if (k <= n) {
            result[idx] = binom_coeff(n, k);
        } else {
            result[idx] = 0;
        }
    }
}

torch::Tensor generate_binom_coeffs_cuda(torch::Tensor L) {
    int batch_size = L.size(0);
    int max_L = torch::max(L).item<int>();
    auto result = torch::zeros({batch_size, max_L + 1}, L.options());

    int threads = 1024;
    int blocks = (batch_size * (max_L + 1) + threads - 1) / threads;

    generateBinomCoeffsKernel<<<blocks, threads>>>(L.data_ptr<float>(), result.data_ptr<float>(), max_L + 1, batch_size);

    return result;
}
