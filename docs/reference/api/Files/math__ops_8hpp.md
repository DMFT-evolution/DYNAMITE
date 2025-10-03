---
title: include/math/math_ops.hpp

---

# include/math/math_ops.hpp



## Functions

|                | Name           |
| -------------- | -------------- |
| template <int P\> <br>constexpr double | **[pow_const](#function-pow-const)**(double q) |
| constexpr double | **[pow_const< 0 >](#function-pow-const<-0->)**(double ) |
| constexpr double | **[pow_const<-1 >](#function-pow-const<-1->)**(double ) |
| double | **[pow_int](#function-pow-int)**(double base, int exp) |
| DMFE_HOST_DEVICE size_t | **[max_device](#function-max-device)**(size_t a, size_t b) |
| DMFE_HOST_DEVICE double | **[max_device](#function-max-device)**(double a, double b) |
| DMFE_HOST_DEVICE size_t | **[min_device](#function-min-device)**(size_t a, size_t b) |
| DMFE_HOST_DEVICE double | **[min_device](#function-min-device)**(double a, double b) |
| double | **[flambda](#function-flambda)**(double q) |
| double | **[Dflambda](#function-dflambda)**(double q) |
| double | **[DDflambda](#function-ddflambda)**(double q) |
| double | **[DDDflambda](#function-dddflambda)**(double q) |


## Functions Documentation

### function pow_const

```cpp
template <int P>
constexpr double pow_const(
    double q
)
```


### function pow_const< 0 >

```cpp
constexpr double pow_const< 0 >(
    double 
)
```


### function pow_const<-1 >

```cpp
constexpr double pow_const<-1 >(
    double 
)
```


### function pow_int

```cpp
double pow_int(
    double base,
    int exp
)
```


### function max_device

```cpp
inline DMFE_HOST_DEVICE size_t max_device(
    size_t a,
    size_t b
)
```


### function max_device

```cpp
inline DMFE_HOST_DEVICE double max_device(
    double a,
    double b
)
```


### function min_device

```cpp
inline DMFE_HOST_DEVICE size_t min_device(
    size_t a,
    size_t b
)
```


### function min_device

```cpp
inline DMFE_HOST_DEVICE double min_device(
    double a,
    double b
)
```


### function flambda

```cpp
double flambda(
    double q
)
```


### function Dflambda

```cpp
double Dflambda(
    double q
)
```


### function DDflambda

```cpp
double DDflambda(
    double q
)
```


### function DDDflambda

```cpp
double DDDflambda(
    double q
)
```




## Source code

```cpp
// Basic math helper utilities (host + device)
#pragma once
#include "core/config_build.hpp"
#include <cstddef>

#if DMFE_WITH_CUDA
#include <cuda_runtime.h>
#endif

// Host compile-time power
template<int P>
constexpr double pow_const(double q) { return q * pow_const<P-1>(q); }
template<> constexpr double pow_const<0>(double) { return 1.0; }
template<> constexpr double pow_const<-1>(double) { return 0.0; }

#if DMFE_WITH_CUDA
// Device version
template<int P>
__device__ __forceinline__ double pow_const_device(double q) { return q * pow_const_device<P-1>(q); }
template<> __device__ __forceinline__ double pow_const_device<0>(double) { return 1.0; }
template<> __device__ __forceinline__ double pow_const_device<-1>(double) { return 0.0; }

// Integer power (host) and fast device integer power
double pow_int(double base, int exp);
__device__ __forceinline__ double fast_pow_int(double base, int exp);
#else
// CPU-only version
double pow_int(double base, int exp);
#endif

// Min / Max helpers (host + device)
DMFE_HOST_DEVICE inline size_t max_device(size_t a, size_t b) { return (a > b) ? a : b; }
DMFE_HOST_DEVICE inline double max_device(double a, double b) { return (a > b) ? a : b; }
DMFE_HOST_DEVICE inline size_t min_device(size_t a, size_t b) { return (a < b) ? a : b; }
DMFE_HOST_DEVICE inline double min_device(double a, double b) { return (a < b) ? a : b; }

// Lambda polynomial helpers (host)
double flambda(double q);
double Dflambda(double q);
double DDflambda(double q);
double DDDflambda(double q);

#if DMFE_WITH_CUDA
// Device versions
__device__ double flambdaGPU(double q);
__device__ double DflambdaGPU(double q);
__device__ double DflambdaGPU2(double q);
__device__ double DDflambdaGPU(double q);
__device__ double DDDflambdaGPU(double q);
#endif
```


-------------------------------

Updated on 2025-10-03 at 23:06:53 +0200
