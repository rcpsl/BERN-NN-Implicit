#include <array>
#include <iostream>
#include <limits>
#include <cassert>
#include <cstdio>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE_X = 32;
constexpr int BLOCK_SIZE_Y = 32;
constexpr int WARP_SIZE = 32;

template<typename scalar_t, int dim>
using packed_accessor_t = torch::PackedTensorAccessor64<scalar_t, dim, torch::RestrictPtrTraits>;

/**
 * This performs a reduction of data in a single GPU warp.
 * `op` can be any binary operation. 
 * The result of the reduction is stored in block[0]
 * This is based on these slides:
 * https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 */
#define WARP_REDUCE(name,op)                                                          \
template<typename scalar_t, int block_size>                                           \
__device__                                                                            \
scalar_t name(volatile scalar_t* block, int local_tid) { 			      \
  if (block_size >= 64) block[local_tid] = op(block[local_tid], block[local_tid+32]); \
  if (block_size >= 32) block[local_tid] = op(block[local_tid], block[local_tid+16]); \
  if (block_size >= 16) block[local_tid] = op(block[local_tid], block[local_tid+8]);  \
  if (block_size >=  8) block[local_tid] = op(block[local_tid], block[local_tid+4]);  \
  if (block_size >=  4) block[local_tid] = op(block[local_tid], block[local_tid+2]);  \
  if (block_size >=  2) block[local_tid] = op(block[local_tid], block[local_tid+1]);  \
}

WARP_REDUCE(warp_min, std::min)
WARP_REDUCE(warp_max, std::max)

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
	const int64_t ebf_size) { 
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;

  // global x idx determines the term.
  int term_id = blockIdx.x;

  while (term_id < nterms) {
    int ebf_id = threadIdx.x;
    while (ebf_id < ebf_size) {
      // ebf_id corresponds to a set of indices {i, j, k, ...}
      // ebf_id = i * d^{v-1} + j * d^{v-2} + k * d^{v-3} + ...
      // So, i = floor(ebf_id / d^{v-1})           (the lower order terms removed by floor)
      // Then, j = floor(ebf_id / d^{v-2}) - i * d
      // THen, k = floor(ebf_id / d^{v-3}) - i * d^2 - j * d
      scalar_t accum_prod = 1;
      int tracker = 0;
      for (int v = 0; v < nvars; ++v) {
        int64_t index = (ebf_id / int_pow(max_degree, nvars - v - 1)) - tracker;
	accum_prod *= poly[term_id][v][index];
	tracker += index;
	tracker *= max_degree;
      } 
      block_ebf_sum[blockIdx.x][ebf_id] += accum_prod;
      ebf_id += blockDim.x;
    } 
    term_id += gridDim.x;
  }

  //__syncthreads();

  //int col_id = threadIdx.x;
  //while (col_id < ebf_size) {
  //  col_id += blockDim.x;
  //}
}

/**
 * Compute the min/max of a Bernstein polynomial in implicit form.
 */ 
torch::Tensor ibf_minmax_cuda(torch::Tensor poly) {
  int nterms = poly.size(0);
  int nvars = poly.size(1);
  int max_degree = poly.size(2);
  int64_t ebf_size = int_pow(max_degree, nvars);

  int64_t target_blocks_x = nterms;
  int64_t blocks_x = std::min(target_blocks_x, static_cast<int64_t>(int_pow(2, 8)));

  auto options = torch::TensorOptions()
	  .dtype(poly.dtype())
	  .device(poly.device());

  auto block_ebf_sum = torch::zeros({blocks_x, ebf_size}, options);

  int blocks = blocks_x;
  int threads = BLOCK_SIZE_X;

  AT_DISPATCH_FLOATING_TYPES(poly.type(), "ibf_cuda_minmax", ([&] {
    ibf_minmax_cuda_kernel<scalar_t><<<blocks, threads>>>(
      block_ebf_sum.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
      poly.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
      nterms,
      nvars,
      max_degree,
      ebf_size);
  }));

  auto ebf_sum = block_ebf_sum.sum(/*dim=*/0);

  auto minmax = torch::empty(2, options);
  minmax[0] = std::get<0>(ebf_sum.min(0)).item();
  minmax[1] = std::get<0>(ebf_sum.max(0)).item();
  return minmax;
}
