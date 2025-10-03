---
title: include/interpolation/interpolation_core.hpp

---

# include/interpolation/interpolation_core.hpp



## Functions

|                | Name           |
| -------------- | -------------- |
| void | **[interpolate](#function-interpolate)**(const std::vector< double > & posB1xIn ={}, const std::vector< double > & posB2xIn ={}, const bool same =false) |


## Functions Documentation

### function interpolate

```cpp
void interpolate(
    const std::vector< double > & posB1xIn ={},
    const std::vector< double > & posB2xIn ={},
    const bool same =false
)
```




## Source code

```cpp
#ifndef INTERPOLATION_CORE_HPP
#define INTERPOLATION_CORE_HPP

#include "core/config_build.hpp"
#include <vector>
#include "core/stream_pool.hpp"

#if DMFE_WITH_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#endif

// CPU interpolation function
void interpolate(const std::vector<double>& posB1xIn = {}, 
                 const std::vector<double>& posB2xIn = {},
                 const bool same = false);

#if DMFE_WITH_CUDA
// GPU interpolation function
void interpolateGPU(const double* posB1xIn = nullptr,
                    const double* posB2xIn = nullptr,
                    const bool same = false,
                    StreamPool* pool = nullptr);

// Helper functions
void diffNfloor(const thrust::device_vector<double>& posB1x,
                thrust::device_vector<size_t>& Floor,
                thrust::device_vector<double>& diff,
                cudaStream_t stream = 0);

// CUDA kernel
__global__ void diffNfloorKernel(const double* __restrict__ posB1x,
                                 size_t* __restrict__ Floor,
                                 double* __restrict__ diff,
                                 size_t len);
#endif // DMFE_WITH_CUDA

#endif // INTERPOLATION_CORE_HPP
```


-------------------------------

Updated on 2025-10-03 at 23:06:53 +0200
