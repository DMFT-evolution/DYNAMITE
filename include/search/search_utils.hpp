#ifndef SEARCH_UTILS_HPP
#define SEARCH_UTILS_HPP

#include "core/config_build.hpp"
#include <vector>

#if DMFE_WITH_CUDA
#include <cuda_runtime.h>
#include "core/device_vector.hpp"
#endif

// CPU binary search function
std::vector<double> bsearchPosSorted(const std::vector<double>& list, const std::vector<double>& elem);

#if DMFE_WITH_CUDA
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
#endif

// CPU interpolation search with initial values
std::vector<double> isearchPosSortedInit(const std::vector<double>& list, const std::vector<double>& elem, const std::vector<double>& inits);

#if DMFE_WITH_CUDA
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

#endif // SEARCH_UTILS_HPP
