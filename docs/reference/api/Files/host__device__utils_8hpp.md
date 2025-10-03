---
title: include/core/host_device_utils.hpp

---

# include/core/host_device_utils.hpp



## Functions

|                | Name           |
| -------------- | -------------- |
| void | **[Product](#function-product)**(const std::vector< double > & a, const std::vector< double > & b, std::vector< double > & out) |
| void | **[scaleVec](#function-scalevec)**(const std::vector< double > & in, double s, std::vector< double > & out) |
| void | **[printVectorDifference](#function-printvectordifference)**(const std::vector< double > & a, const std::vector< double > & b) |
| thrust::device_vector< double > | **[SubtractGPU](#function-subtractgpu)**(const thrust::device_vector< double > & a, const thrust::device_vector< double > & b) |
| thrust::device_vector< double > | **[scalarMultiply](#function-scalarmultiply)**(const thrust::device_vector< double > & v, double s) |
| thrust::device_vector< double > | **[scalarMultiply_ptr](#function-scalarmultiply-ptr)**(const thrust::device_ptr< double > & v, double s, size_t len) |
| void | **[AddSubtractGPU](#function-addsubtractgpu)**(thrust::device_vector< double > & gK, const thrust::device_vector< double > & gKfinal, const thrust::device_vector< double > & gK0, thrust::device_vector< double > & gR, const thrust::device_vector< double > & gRfinal, const thrust::device_vector< double > & gR0, cudaStream_t stream =0) |
| void | **[FusedUpdate](#function-fusedupdate)**(const thrust::device_ptr< double > & a, const thrust::device_ptr< double > & b, const thrust::device_vector< double > & out, const double * alpha, const double * beta, const thrust::device_vector< double > * delta =nullptr, const thrust::device_vector< double > * extra1 =nullptr, const thrust::device_vector< double > * extra2 =nullptr, const thrust::device_vector< double > * extra3 =nullptr, const thrust::device_ptr< double > & subtract =nullptr, cudaStream_t stream =0) |


## Functions Documentation

### function Product

```cpp
void Product(
    const std::vector< double > & a,
    const std::vector< double > & b,
    std::vector< double > & out
)
```


### function scaleVec

```cpp
void scaleVec(
    const std::vector< double > & in,
    double s,
    std::vector< double > & out
)
```


### function printVectorDifference

```cpp
void printVectorDifference(
    const std::vector< double > & a,
    const std::vector< double > & b
)
```


### function SubtractGPU

```cpp
thrust::device_vector< double > SubtractGPU(
    const thrust::device_vector< double > & a,
    const thrust::device_vector< double > & b
)
```


### function scalarMultiply

```cpp
thrust::device_vector< double > scalarMultiply(
    const thrust::device_vector< double > & v,
    double s
)
```


### function scalarMultiply_ptr

```cpp
thrust::device_vector< double > scalarMultiply_ptr(
    const thrust::device_ptr< double > & v,
    double s,
    size_t len
)
```


### function AddSubtractGPU

```cpp
void AddSubtractGPU(
    thrust::device_vector< double > & gK,
    const thrust::device_vector< double > & gKfinal,
    const thrust::device_vector< double > & gK0,
    thrust::device_vector< double > & gR,
    const thrust::device_vector< double > & gRfinal,
    const thrust::device_vector< double > & gR0,
    cudaStream_t stream =0
)
```


### function FusedUpdate

```cpp
void FusedUpdate(
    const thrust::device_ptr< double > & a,
    const thrust::device_ptr< double > & b,
    const thrust::device_vector< double > & out,
    const double * alpha,
    const double * beta,
    const thrust::device_vector< double > * delta =nullptr,
    const thrust::device_vector< double > * extra1 =nullptr,
    const thrust::device_vector< double > * extra2 =nullptr,
    const thrust::device_vector< double > * extra3 =nullptr,
    const thrust::device_ptr< double > & subtract =nullptr,
    cudaStream_t stream =0
)
```




## Source code

```cpp
#pragma once
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <cassert>

// Host utilities
// Element-wise product: out[i] = a[i] * b[i]
void Product(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out);
void scaleVec(const std::vector<double>& in, double s, std::vector<double>& out);
void printVectorDifference(const std::vector<double>& a, const std::vector<double>& b);

// Device utilities
thrust::device_vector<double> SubtractGPU(const thrust::device_vector<double>& a, const thrust::device_vector<double>& b);
thrust::device_vector<double> scalarMultiply(const thrust::device_vector<double>& v, double s);
thrust::device_vector<double> scalarMultiply_ptr(const thrust::device_ptr<double>& v, double s, size_t len);

void AddSubtractGPU(thrust::device_vector<double>& gK,
                    const thrust::device_vector<double>& gKfinal,
                    const thrust::device_vector<double>& gK0,
                    thrust::device_vector<double>& gR,
                    const thrust::device_vector<double>& gRfinal,
                    const thrust::device_vector<double>& gR0,
                    cudaStream_t stream = 0);

void FusedUpdate(const thrust::device_ptr<double>& a,
                 const thrust::device_ptr<double>& b,
                 const thrust::device_vector<double>& out,
                 const double* alpha,
                 const double* beta,
                 const thrust::device_vector<double>* delta = nullptr,
                 const thrust::device_vector<double>* extra1 = nullptr,
                 const thrust::device_vector<double>* extra2 = nullptr,
                 const thrust::device_vector<double>* extra3 = nullptr,
                 const thrust::device_ptr<double>& subtract = nullptr,
                 cudaStream_t stream = 0);
```


-------------------------------

Updated on 2025-10-03 at 23:06:52 +0200
