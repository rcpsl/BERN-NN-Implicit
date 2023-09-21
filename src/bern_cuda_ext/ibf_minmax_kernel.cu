#include <array>
#include <iostream>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_VARS = 20;

template<typename scalar_t, int dim>
using packed_accessor_t = torch::PackedTensorAccessor64<scalar_t, dim, torch::RestrictPtrTraits>;

/**
 * This performs a reduction of data in a single GPU warp.
 * `op` can be any binary operation. 
 * The result of the reduction is stored in block[local_tid]
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


// TODO: CUDA doesn't provide an int-only pow?
// Casting int to double to int may be undesirable.
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
	packed_accessor_t<scalar_t, 1> local_min,
	packed_accessor_t<scalar_t, 1> local_max,
	const packed_accessor_t<scalar_t, 3> poly,
	const int nterms,
	const int nvars,
	const int max_degree,
	const int ebf_size) {
  size_t local_tid = threadIdx.x;
  size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  __shared__ scalar_t block_outer[BLOCK_SIZE];
  __shared__ scalar_t block_min[BLOCK_SIZE];
  __shared__ scalar_t block_max[BLOCK_SIZE];

  block_outer[local_tid] = 0;

  if (global_tid < ebf_size) {
    
    // Step 1: compute sum of outer products for each term.
    for (int term = 0; term < nterms; ++term) {
      // each global_tid must correspond to a unique sequence of powers in
      // each variable. To do this, we follow the same iteration as converting
      // an base 10 integer t to an integer of base `max_degree`.
      // t % d is least significant digit in base d.
      // t / d truncates the last digit in base d.
      //
      // Since each term is a tensor of the same size and shape,
      // we can reuse the same indexing scheme for each term.
      scalar_t local_result = 1;
      int t = global_tid;
      for (int var = 0; var < nvars; ++var) {
        int pow_idx = t % max_degree;
        t /= max_degree;
        local_result *= poly[term][var][pow_idx];
      }
      block_outer[local_tid] += local_result;
    }

    block_min[local_tid] = block_outer[local_tid];
    block_max[local_tid] = block_outer[local_tid];

    __syncthreads();

    // Step 2: Find the min and max for each CUDA block.
    for (int s = blockDim.x/2; s > 32; s >>= 1) {
      if (local_tid < s) {
        block_min[local_tid] = std::min(block_min[local_tid], block_min[local_tid + s]);
        block_max[local_tid] = std::max(block_max[local_tid], block_max[local_tid + s]);
      }
    }

    if (local_tid < 32) warp_min<scalar_t, BLOCK_SIZE>(block_min, local_tid);
    if (local_tid < 32) warp_max<scalar_t, BLOCK_SIZE>(block_max, local_tid);

    local_min[blockIdx.x] = block_min[0];
    local_max[blockIdx.x] = block_max[0];
  }
}

/**
 * Compute the min/max of a Bernstein polynomial in implicit form.
 */ 
torch::Tensor ibf_minmax_cuda(torch::Tensor poly) {
  int nterms = poly.size(0);
  int nvars = poly.size(1);
  int max_degree = poly.size(2);

  int64_t ebf_size = int_pow(nvars, max_degree);
  int threads = BLOCK_SIZE;
  int blocks = static_cast<int>(std::ceil(static_cast<float>(ebf_size) /
			                  static_cast<float>(threads)));

  auto block_min = torch::zeros(blocks).to(poly.device());
  auto block_max = torch::zeros(blocks).to(poly.device());

  AT_DISPATCH_FLOATING_TYPES(poly.type(), "ibf_cuda_minmax", ([&] {
    ibf_minmax_cuda_kernel<scalar_t><<<blocks, threads>>>(
      block_min.packed_accessor64<scalar_t,1,torch::RestrictPtrTraits>(),
      block_max.packed_accessor64<scalar_t,1,torch::RestrictPtrTraits>(),
      poly.packed_accessor64<scalar_t,3,torch::RestrictPtrTraits>(),
      nterms,
      nvars,
      max_degree,
      ebf_size);
  }));

  auto minmax = torch::empty(2).to(poly.device());
  minmax[0] = std::get<0>(block_min.min(0)).item();
  minmax[1] = std::get<0>(block_max.max(0)).item();

  return minmax;
}
