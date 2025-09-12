// compute_utils.cu - Definitions of basic element-wise CUDA kernels
#include "compute_utils.hpp"

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
