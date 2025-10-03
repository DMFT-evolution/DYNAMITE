---
title: include/search/search_utils.hpp

---

# include/search/search_utils.hpp



## Functions

|                | Name           |
| -------------- | -------------- |
| std::vector< double > | **[bsearchPosSorted](#function-bsearchpossorted)**(const std::vector< double > & list, const std::vector< double > & elem) |
| std::vector< double > | **[isearchPosSortedInit](#function-isearchpossortedinit)**(const std::vector< double > & list, const std::vector< double > & elem, const std::vector< double > & inits) |


## Functions Documentation

### function bsearchPosSorted

```cpp
std::vector< double > bsearchPosSorted(
    const std::vector< double > & list,
    const std::vector< double > & elem
)
```


### function isearchPosSortedInit

```cpp
std::vector< double > isearchPosSortedInit(
    const std::vector< double > & list,
    const std::vector< double > & elem,
    const std::vector< double > & inits
)
```




## Source code

```cpp
#ifndef SEARCH_UTILS_HPP
#define SEARCH_UTILS_HPP

#include "core/config_build.hpp"
#include <vector>

#if DMFE_WITH_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#endif

// CPU binary search function
std::vector<double> bsearchPosSorted(const std::vector<double>& list, const std::vector<double>& elem);

#if DMFE_WITH_CUDA
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
#endif

// CPU interpolation search with initial values
std::vector<double> isearchPosSortedInit(const std::vector<double>& list, const std::vector<double>& elem, const std::vector<double>& inits);

#if DMFE_WITH_CUDA
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
#endif // DMFE_WITH_CUDA

#endif // SEARCH_UTILS_HPP
```


-------------------------------

Updated on 2025-10-03 at 23:06:53 +0200
