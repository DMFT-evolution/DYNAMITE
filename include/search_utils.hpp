#ifndef SEARCH_UTILS_HPP
#define SEARCH_UTILS_HPP

#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>


// CPU binary search function
std::vector<double> bsearchPosSorted(const std::vector<double>& list, const std::vector<double>& elem);

// GPU binary search functions
thrust::device_vector<double> bsearchPosSortedGPU_slow(
    const thrust::device_vector<double>& list,
    const thrust::device_vector<double>& elem);

thrust::device_vector<double> bsearchPosSortedGPU(
    const thrust::device_vector<double>& list,
    const thrust::device_vector<double>& elem,
    cudaStream_t stream = 0);

void bsearchPosSortedGPU(
    const thrust::device_vector<double>& list,
    const thrust::device_vector<double>& elem,
    thrust::device_vector<double>& result,
    cudaStream_t stream = 0);

// CPU interpolation search with initial values
std::vector<double> isearchPosSortedInit(const std::vector<double>& list, const std::vector<double>& elem, const std::vector<double>& inits);

// GPU interpolation search with initial values
thrust::device_vector<double> isearchPosSortedInitGPU(
    const thrust::device_vector<double>& list,
    const thrust::device_vector<double>& elem,
    const thrust::device_vector<double>& inits);

// CUDA kernel declarations (only for CUDA compilation)
#ifdef __CUDACC__
__global__ __launch_bounds__(64, 1) void bsearch_interp_kernel(
    const double* __restrict__ list,
    const double* __restrict__ elem,
    double* __restrict__ result,
    size_t list_size,
    size_t elem_size);
#endif

#endif // SEARCH_UTILS_HPP
