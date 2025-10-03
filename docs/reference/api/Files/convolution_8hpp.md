---
title: include/convolution/convolution.hpp

---

# include/convolution/convolution.hpp



## Functions

|                | Name           |
| -------------- | -------------- |
| std::vector< double > | **[ConvA](#function-conva)**(const std::vector< double > & f, const std::vector< double > & g, const double t) |
| std::vector< double > | **[ConvR](#function-convr)**(const std::vector< double > & f, const std::vector< double > & g, const double t) |


## Functions Documentation

### function ConvA

```cpp
std::vector< double > ConvA(
    const std::vector< double > & f,
    const std::vector< double > & g,
    const double t
)
```


### function ConvR

```cpp
std::vector< double > ConvR(
    const std::vector< double > & f,
    const std::vector< double > & g,
    const double t
)
```




## Source code

```cpp
#pragma once
#include "core/config_build.hpp"
#include <vector>

#if DMFE_WITH_CUDA
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <cuda_runtime.h>
#endif

// Host convolution functions
std::vector<double> ConvA(const std::vector<double>& f, const std::vector<double>& g, const double t);
std::vector<double> ConvR(const std::vector<double>& f, const std::vector<double>& g, const double t);

#if DMFE_WITH_CUDA
// ConvA (QK-style) GPU interfaces
thrust::device_vector<double> ConvAGPU(const thrust::device_vector<double>& f,
                                       const thrust::device_vector<double>& g,
                                       double t,
                                       const thrust::device_vector<double>& integ,
                                       const thrust::device_vector<double>& theta,
                                       cudaStream_t stream = 0);

thrust::device_vector<double> ConvAGPU(const thrust::device_vector<double>& f,
                                       const thrust::device_ptr<double>& g,
                                       double t,
                                       const thrust::device_vector<double>& integ,
                                       const thrust::device_vector<double>& theta,
                                       cudaStream_t stream = 0);

void ConvAGPU_Stream(const thrust::device_vector<double>& f,
                     const thrust::device_vector<double>& g,
                     thrust::device_vector<double>& out,
                     thrust::device_vector<double>& t,
                     const thrust::device_vector<double>& integ,
                     const thrust::device_vector<double>& theta,
                     cudaStream_t stream = 0);

void ConvAGPU_Stream(const thrust::device_vector<double>& f,
                     const thrust::device_ptr<double>& g,
                     thrust::device_vector<double>& out,
                     double t,
                     const thrust::device_vector<double>& integ,
                     const thrust::device_vector<double>& theta,
                     cudaStream_t stream = 0);

// ConvR (QR-style) GPU interfaces
thrust::device_vector<double> ConvRGPU(const thrust::device_vector<double>& f,
                                       const thrust::device_vector<double>& g,
                                       double t,
                                       const thrust::device_vector<double>& integ,
                                       const thrust::device_vector<double>& theta,
                                       cudaStream_t stream = 0);

void ConvRGPU_Stream(const thrust::device_vector<double>& f,
                     const thrust::device_vector<double>& g,
                     thrust::device_vector<double>& out,
                     const thrust::device_vector<double>& t,
                     const thrust::device_vector<double>& integ,
                     const thrust::device_vector<double>& theta,
                     cudaStream_t stream = 0);
#endif
```


-------------------------------

Updated on 2025-10-03 at 23:06:52 +0200
