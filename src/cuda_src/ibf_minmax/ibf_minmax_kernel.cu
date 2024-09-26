#include <array>
#include <iostream>
#include <limits>
#include <cassert>
#include <cstdio>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;

template<typename scalar_t, int dim>
using packed_accessor_t = torch::PackedTensorAccessor64<scalar_t, dim, torch::RestrictPtrTraits>;

// A table that stores precomputed powers `max_degree ^ x`
// It should be safe to initialize this with 64
// entries because the explicit form is unable to grow
// substantially when the number of variables is large.
constexpr int POW_TABLE_SIZE = 64;
__constant__ int64_t int_pow_table[POW_TABLE_SIZE];

__host__ __device__
int64_t int_pow(int base, int exp) {
  int64_t accum = 1;
  for (int i = 0; i < exp; ++i) {
    accum *= base;
  }
  return accum;
}

template<typename scalar_t>
__global__ 
void ibf_minmax_cuda_kernel(
	packed_accessor_t<scalar_t, 2> block_ebf_sum,
	const packed_accessor_t<scalar_t, 3> poly,
	const int nterms,
	const int nvars,
	const int max_degree,
	const int64_t ebf_chunk_start,
	const int64_t ebf_chunk_end) { 
  int term_id = blockIdx.x;
  while (term_id < nterms) {
    int ebf_id = ebf_chunk_start + threadIdx.x;
    while (ebf_id < ebf_chunk_end) {
      // ebf_id corresponds to a set of indices {i, j, k, ...}
      // ebf_id = i * d^{v-1} + j * d^{v-2} + k * d^{v-3} + ...
      // So, i = floor(ebf_id / d^{v-1})           (the lower order terms removed by floor)
      // Then, j = floor(ebf_id / d^{v-2}) - i * d
      // THen, k = floor(ebf_id / d^{v-3}) - i * d^2 - j * d
      scalar_t accum_prod = 1;
      int tracker = 0;
      for (int v = 0; v < nvars; ++v) {
	int64_t p = int_pow(max_degree, nvars - v - 1);
	//int64_t p = int_pow_table[nvars - v -1];
        int64_t index = (ebf_id / p) - tracker;
	accum_prod *= poly[term_id][v][index];
	tracker += index;
	tracker *= max_degree;
      } 
      block_ebf_sum[blockIdx.x][ebf_id - ebf_chunk_start] += accum_prod;
      ebf_id += blockDim.x;
    } 
    term_id += gridDim.x;
  }
}


/**
 * Compute the min/max of a Bernstein polynomial in implicit form.
 */ 
torch::Tensor ibf_minmax_cuda(torch::Tensor poly) {
  int nterms = poly.size(0);
  int nvars = poly.size(1);
  int max_degree = poly.size(2);
  int64_t ebf_size = int_pow(max_degree, nvars);

  // initialize int_pow_table
  std::vector<int64_t> max_degree_powers(POW_TABLE_SIZE);
  for (int i = 0; i < POW_TABLE_SIZE; ++i) {
    max_degree_powers[i] = int_pow(max_degree, i);
  }
  int64_t* pows = max_degree_powers.data();
  cudaMemcpyToSymbol(int_pow_table, pows, POW_TABLE_SIZE * sizeof(int64_t), 0, cudaMemcpyHostToDevice);

  int64_t target_blocks_x = nterms;
  int64_t blocks_x = std::min(target_blocks_x, static_cast<int64_t>(int_pow(2, 30) / ebf_size));

  auto options = torch::TensorOptions()
    .dtype(poly.dtype())
    .device(poly.device());

  //int64_t ebf_chunk_size = int_pow(2, 6);
  //int64_t num_ebf_chunks = std::ceil(static_cast<float>(ebf_size) /
		 		     //static_cast<float>(ebf_chunk_size));

  //auto mins = torch::full(num_ebf_chunks, 99999, options);
  //auto maxs = torch::full(num_ebf_chunks, -99999, options);

  /*for (int ebf_chunk_start = 0; ebf_chunk_start < ebf_size; ebf_chunk_start += ebf_chunk_size) {
    int ebf_chunk_end = std::min(ebf_size, ebf_chunk_start + ebf_chunk_size);
    auto block_ebf_sum = torch::zeros({blocks_x, ebf_chunk_end - ebf_chunk_start}, options);
    const int blocks = blocks_x;
    const int threads = BLOCK_SIZE;
    AT_DISPATCH_FLOATING_TYPES(poly.type(), "ibf_cuda_minmax", ([&] {
      ibf_minmax_cuda_kernel<scalar_t><<<blocks, threads>>>(
        block_ebf_sum.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        poly.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
        nterms,
        nvars,
        max_degree,
	ebf_chunk_start,
	ebf_chunk_end);
    }));
    */
  auto ebf_sum = torch::zeros({blocks_x, ebf_size}, options);
  const int threads = BLOCK_SIZE;
  
  AT_DISPATCH_FLOATING_TYPES(poly.type(), "ibf_cuda_minmax", ([&] {
    ibf_minmax_cuda_kernel<scalar_t><<<blocks_x, threads>>>(
      ebf_sum.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
      poly.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
      nterms,
      nvars,
      max_degree,
      0,
      ebf_size);
  }));

  auto sum = ebf_sum.sum(/*dim=*/0);
  auto minmax = torch::empty(2, options);
  minmax[0] = sum.min();
  minmax[1] = sum.max();

  return minmax;
}
