#include <array>
#include <iostream>
#include <limits>
#include <cassert>
#include <cstdio>
#include <algorithm>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 128;

template<typename scalar_t, int dim>
using packed_accessor_t = torch::PackedTensorAccessor64<scalar_t, dim, torch::RestrictPtrTraits>;

template<typename scalar_t>
__global__ 
void quadrant_ibf_minmax_cuda_kernel(
	packed_accessor_t<scalar_t, 1> term_min,
	packed_accessor_t<scalar_t, 1> term_max,
	const packed_accessor_t<scalar_t, 3> poly,
	const packed_accessor_t<int64_t, 2> first_nonzero,
	const packed_accessor_t<int64_t, 2> last_nonzero,
	const int nterms,
	const int nvars,
	const int max_degree) { 
  scalar_t thread_min = 0; 
  scalar_t thread_max = 0; 

  const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int term_id = thread_id;
  while (term_id < nterms) {
    scalar_t accum_top_left = 1;
    scalar_t accum_bottom_right = 1;
    for (int v = 0; v < nvars; ++v) {
      accum_top_left *= poly[term_id][v][first_nonzero[term_id][v]];
      accum_bottom_right *= poly[term_id][v][last_nonzero[term_id][v]];
    }
    thread_min += std::min(accum_top_left, accum_bottom_right);
    thread_max += std::max(accum_top_left, accum_bottom_right);
    term_id += blockDim.x * gridDim.x;
  }

  term_min[thread_id] = thread_min;
  term_max[thread_id] = thread_max; 
}

/**
 * Compute the min/max of a Bernstein polynomial in implicit form.
 */ 
torch::Tensor quadrant_ibf_minmax_cuda(torch::Tensor poly) {
  int64_t nterms = poly.size(0);
  int64_t nvars = poly.size(1);
  int64_t max_degree = poly.size(2);

  auto options = torch::TensorOptions()
    .dtype(poly.dtype())
    .device(poly.device());

  auto nonzero_mask = (poly != 0).to(torch::kInt64);
  auto first_nonzero = nonzero_mask.argmax(2);
  auto last_nonzero = nonzero_mask.size(2) - 1 - nonzero_mask.flip({2}).argmax(2);

  int64_t target_blocks_x = (nterms + BLOCK_SIZE) / BLOCK_SIZE;
  int64_t blocks_x = std::min(target_blocks_x, static_cast<int64_t>(std::pow(2.0, 30)));
  int64_t blocks = blocks_x;
  const int threads = BLOCK_SIZE;

  auto term_min = torch::zeros(std::min((int64_t)blocks * threads, nterms), options);
  auto term_max = torch::zeros(std::min((int64_t)blocks * threads, nterms), options);
  
  AT_DISPATCH_FLOATING_TYPES(poly.type(), "ibf_cuda_minmax", ([&] {
      quadrant_ibf_minmax_cuda_kernel<scalar_t><<<blocks, threads>>>(
        term_min.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
        term_max.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
        poly.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
        first_nonzero.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
        last_nonzero.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
        nterms,
        nvars,
        max_degree);
    }));

  auto minmax = torch::empty(2, options);
  minmax[0] = term_min.sum();
  minmax[1] = term_max.sum();

  return minmax;
}
