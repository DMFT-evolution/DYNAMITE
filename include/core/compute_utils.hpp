// compute_utils.hpp - Basic element-wise CUDA kernels
#pragma once
#include "core/config_build.hpp"
#include <cstddef>

#if DMFE_WITH_CUDA
#include <cuda_runtime.h>
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

// ============================================================================
// [PERF] Batched forms of the two element-wise kernels the Runge-Kutta drivers
// launch several times per stage (up to four computeMA and two
// computeWeightedSum).  Each entry k of the batch is handled by blockIdx.y == k
// and executes the body of the single kernel unchanged.
//
// BIT-EXACTNESS: element idx of entry k is computed by exactly the same
// expression from exactly the same operands -- in computeWeightedSum the inner
// accumulation over the stages keeps its original order -- and no thread reads
// what another thread writes.  Only the number of launches changes.
// ============================================================================
struct MABatchDesc {
    static constexpr int MAX = 8;
    double*       a[MAX];
    const double* b[MAX];
    double        c[MAX];
    int           n = 0;
    void add(double* a_, const double* b_, double c_) { a[n] = a_; b[n] = b_; c[n] = c_; ++n; }
};
void computeMABatchedGPU(const MABatchDesc& batch, size_t N, cudaStream_t stream);

struct WSumBatchDesc {
    static constexpr int MAX = 2;
    const double* g0[MAX];
    const double* h[MAX];
    double*       out[MAX];
    int           n = 0;
    void add(const double* g0_, const double* h_, double* out_) { g0[n] = g0_; h[n] = h_; out[n] = out_; ++n; }
};
void computeWeightedSumBatchedGPU(const WSumBatchDesc& batch, const double* a,
                                  double dt, int nstages, size_t len, cudaStream_t stream);
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
#endif
