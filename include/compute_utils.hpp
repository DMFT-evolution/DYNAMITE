// compute_utils.hpp - Basic element-wise CUDA kernels
#pragma once
#include <cstddef>

// Scaling: a[i] *= b
__global__ void computeScale(double* a, double b, size_t N);
// Product: a[i] *= b[i]
__global__ void computeProduct(double* a, const double* b, size_t N);
// Sum in-place: a[i] += b[i]
__global__ void computeSum(double* a, const double* b, size_t N);
// Sum out-of-place: c[i] = a[i] + b[i]
__global__ void computeSum(const double* a, const double* b, double* c, size_t N);
// Diff in-place: a[i] -= b[i]
__global__ void computeDiff(double* a, const double* b, size_t N);
// Diff out-of-place: c[i] = a[i] - b[i]
__global__ void computeDiff(const double* a, const double* b, double* c, size_t N);
// Multiply-add: a[i] += b[i] * c
__global__ void computeMA(double* a, const double* b, double c, size_t N);
// MAD variant: a[i] = 2*a[i] - b[i] + c[i] * d
__global__ void computeMAD(double* a, const double* b, const double* c, double d, size_t N);
// Weighted sum across n rows: result[j] = gK0[j] + dt * sum_i a[i]*hK[i*len + j]
__global__ void computeWeightedSum(const double* __restrict__ gK0,
                                   const double* __restrict__ hK,
                                   const double* __restrict__ a,
                                   double* __restrict__ result,
                                   double dt,
                                   int n,
                                   size_t len);
