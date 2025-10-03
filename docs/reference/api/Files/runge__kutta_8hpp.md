---
title: include/EOMs/runge_kutta.hpp

---

# include/EOMs/runge_kutta.hpp



## Functions

|                | Name           |
| -------------- | -------------- |
| double | **[SSPRK104](#function-ssprk104)**() |
| double | **[RK54](#function-rk54)**() |
| void | **[init_RK54GPU](#function-init-rk54gpu)**() |
| void | **[init_SSPRK104GPU](#function-init-ssprk104gpu)**() |
| void | **[init_SERK2](#function-init-serk2)**(int q) |
| double | **[update](#function-update)**() |
| long double | **[chebyshevT_ld](#function-chebyshevt-ld)**(int n, long double x) |
| long double | **[chebyshevU_ld](#function-chebyshevu-ld)**(int n, long double x) |
| std::vector< long double > | **[gaussianElimination_ld](#function-gaussianelimination-ld)**(std::vector< std::vector< long double > > A, std::vector< long double > b) |
| std::vector< double > | **[SERKcoeffs](#function-serkcoeffs)**(int q) |


## Functions Documentation

### function SSPRK104

```cpp
double SSPRK104()
```


### function RK54

```cpp
double RK54()
```


### function init_RK54GPU

```cpp
void init_RK54GPU()
```


### function init_SSPRK104GPU

```cpp
void init_SSPRK104GPU()
```


### function init_SERK2

```cpp
void init_SERK2(
    int q
)
```


### function update

```cpp
double update()
```


### function chebyshevT_ld

```cpp
long double chebyshevT_ld(
    int n,
    long double x
)
```


### function chebyshevU_ld

```cpp
long double chebyshevU_ld(
    int n,
    long double x
)
```


### function gaussianElimination_ld

```cpp
std::vector< long double > gaussianElimination_ld(
    std::vector< std::vector< long double > > A,
    std::vector< long double > b
)
```


### function SERKcoeffs

```cpp
std::vector< double > SERKcoeffs(
    int q
)
```




## Source code

```cpp
#ifndef RUNGE_KUTTA_HPP
#define RUNGE_KUTTA_HPP

#include "core/config_build.hpp"
#include <vector>
#include "core/stream_pool.hpp"

#if DMFE_WITH_CUDA
// Forward declarations for kernel functions used by Runge-Kutta methods
__global__ void computeError(const double* __restrict__ gKfinal,
                            const double* __restrict__ gKe,
                            const double* __restrict__ gRfinal,
                            const double* __restrict__ gRe,
                            double* __restrict__ result,
                            size_t len);
#endif

// CPU Runge-Kutta methods
double SSPRK104();
double RK54();

// Runge-Kutta initialization functions (work for both CPU and GPU)
void init_RK54GPU();
void init_SSPRK104GPU();
void init_SERK2(int q);

double update();

#if DMFE_WITH_CUDA
// GPU Runge-Kutta methods
double RK54GPU(StreamPool* pool = nullptr);
double SSPRK104GPU(StreamPool* pool = nullptr);
double SERK2GPU(int q, StreamPool* pool = nullptr);

// Helper functions for method selection
double updateGPU(StreamPool* pool = nullptr);
#endif

// SERK coefficient generation functions
long double chebyshevT_ld(int n, long double x);
long double chebyshevU_ld(int n, long double x);
std::vector<long double> gaussianElimination_ld(std::vector<std::vector<long double>> A, std::vector<long double> b);
std::vector<double> SERKcoeffs(int q);

#endif // RUNGE_KUTTA_HPP
```


-------------------------------

Updated on 2025-10-03 at 23:06:53 +0200
