// compute_utils.cu - Definitions of basic element-wise CUDA kernels
#include "core/compute_utils.hpp"

__global__ void computeScale(double* a, double b, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) a[idx] *= b;
}

__global__ void computeProduct(double* a, const double* b, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) a[idx] *= b[idx];
}

__global__ void computeSum(double* a, const double* b, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) a[idx] += b[idx];
}

__global__ void computeSum(const double* a, const double* b, double* c, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

__global__ void computeDiff(double* a, const double* b, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) a[idx] -= b[idx];
}

__global__ void computeDiff(const double* a, const double* b, double* c, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] - b[idx];
}

__global__ void computeMA(double* a, const double* b, double c, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) a[idx] += b[idx] * c;
}

// [PERF] see core/compute_utils.hpp
__global__ void computeMABatchedKernel(MABatchDesc batch, size_t N) {
    const int k = blockIdx.y;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) batch.a[k][idx] += batch.b[k][idx] * batch.c[k];
}

void computeMABatchedGPU(const MABatchDesc& batch, size_t N, cudaStream_t stream) {
    if (batch.n == 0) return;
    const int threads = 64;                        // must match the single-kernel value
    const int blocks  = static_cast<int>((N + threads - 1) / threads);
    dim3 grid(static_cast<unsigned>(blocks), static_cast<unsigned>(batch.n));
    computeMABatchedKernel<<<grid, threads, 0, stream>>>(batch, N);
}

__global__ void computeWeightedSumBatchedKernel(
    WSumBatchDesc batch,
    const double* __restrict__ a,
    double dt,
    int n,
    size_t len) {
    const int k = blockIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= len) return;
    const double* __restrict__ hK = batch.h[k];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += dt * a[i] * hK[i * len + j];
    }
    batch.out[k][j] = batch.g0[k][j] + sum;
}

void computeWeightedSumBatchedGPU(const WSumBatchDesc& batch, const double* a,
                                  double dt, int nstages, size_t len, cudaStream_t stream) {
    if (batch.n == 0) return;
    const int threads = 64;                        // must match the single-kernel value
    const int blocks  = static_cast<int>((len + threads - 1) / threads);
    dim3 grid(static_cast<unsigned>(blocks), static_cast<unsigned>(batch.n));
    computeWeightedSumBatchedKernel<<<grid, threads, 0, stream>>>(batch, a, dt, nstages, len);
}

__global__ void computeMAD(double* a, const double* b, const double* c, double d, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) a[idx] = 2 * a[idx] - b[idx] + c[idx] * d;
}

__global__ void computeWeightedSum(
    const double* __restrict__ gK0,
    const double* __restrict__ hK,
    const double* __restrict__ a,
    double* __restrict__ result,
    double dt,
    int n,
    size_t len) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= len) return;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += dt * a[i] * hK[i * len + j];
    }
    result[j] = gK0[j] + sum;
}
