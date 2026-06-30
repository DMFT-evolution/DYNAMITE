#ifndef SEARCH_UTILS_CUH
#define SEARCH_UTILS_CUH

#include "search/search_utils.hpp"

#if DMFE_WITH_CUDA
#include <cuda_runtime.h>
#include "core/device_vector.hpp"


// GPU binary search functions
dmfe::device_vector<double> bsearchPosSortedGPU_slow(
    const dmfe::device_vector<double>& list,
    const dmfe::device_vector<double>& elem);

dmfe::device_vector<double> bsearchPosSortedGPU(
    const dmfe::device_vector<double>& list,
    const dmfe::device_vector<double>& elem,
    cudaStream_t stream = 0);

void bsearchPosSortedGPU(
    const dmfe::device_vector<double>& list,
    const dmfe::device_vector<double>& elem,
    dmfe::device_vector<double>& result,
    cudaStream_t stream = 0);

    
// GPU interpolation search with initial values
dmfe::device_vector<double> isearchPosSortedInitGPU(
    const dmfe::device_vector<double>& list,
    const dmfe::device_vector<double>& elem,
    const dmfe::device_vector<double>& inits);

// CUDA kernel declarations (only for CUDA compilation)
#ifdef __CUDACC__
__global__ __launch_bounds__(64, 1) void bsearch_interp_kernel(
    const double* __restrict__ list,
    const double* __restrict__ elem,
    double* __restrict__ result,
    size_t list_size,
    size_t elem_size);
#endif
#endif // DMFE_WITH_CUDA

#endif // SEARCH_UTILS_CUH
