// Basic math helper utilities (host + device)
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

// Host compile-time power
template<int P>
constexpr double pow_const(double q) { return q * pow_const<P-1>(q); }
template<> constexpr double pow_const<0>(double) { return 1.0; }
template<> constexpr double pow_const<-1>(double) { return 0.0; }

// Device version
template<int P>
__device__ __forceinline__ double pow_const_device(double q) { return q * pow_const_device<P-1>(q); }
template<> __device__ __forceinline__ double pow_const_device<0>(double) { return 1.0; }
template<> __device__ __forceinline__ double pow_const_device<-1>(double) { return 0.0; }

// Integer power (host) and fast device integer power
double pow_int(double base, int exp);
__device__ __forceinline__ double fast_pow_int(double base, int exp);

// Min / Max helpers (host + device)
__device__ __host__ inline size_t max_device(size_t a, size_t b) { return (a > b) ? a : b; }
__device__ __host__ inline double max_device(double a, double b) { return (a > b) ? a : b; }
__device__ __host__ inline size_t min_device(size_t a, size_t b) { return (a < b) ? a : b; }
__device__ __host__ inline double min_device(double a, double b) { return (a < b) ? a : b; }

// Lambda polynomial helpers (host)
double flambda(double q);
double Dflambda(double q);
double DDflambda(double q);
double DDDflambda(double q);

// Device versions
__device__ double flambdaGPU(double q);
__device__ double DflambdaGPU(double q);
__device__ double DflambdaGPU2(double q);
__device__ double DDflambdaGPU(double q);
__device__ double DDDflambdaGPU(double q);

