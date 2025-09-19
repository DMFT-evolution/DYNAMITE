#include "index_mat.hpp"
#include "config.hpp"
#include "simulation_data.hpp"
#include <algorithm>
#include <numeric>
// #include <thrust/execution_policy.h>
// #include <cuda_runtime.h>

using namespace std;

// External declarations for global variables
extern SimulationConfig config;
extern SimulationData* sim;

// 2D interpolation of fields (QK/QR) on a staggered grid. Provides GPU kernels and CPU fallback.

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ inline double atomicAdd_double(double* address, double val) {
    unsigned long long* addr = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *addr, assumed;
    do {
        assumed = old;
        double sum = __longlong_as_double(assumed) + val;
        old = atomicCAS(addr, assumed, __double_as_longlong(sum));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#define ATOMIC_ADD_DBL(addr, v) atomicAdd_double(addr, v)
#else
#define ATOMIC_ADD_DBL(addr, v) atomicAdd(addr, v)
#endif

// Dynamic-depth kernel: supports arbitrary stencil depth at runtime
__global__ __launch_bounds__(64, 1) void indexMatAllKernel_dynamic(const double* __restrict__ posx,
                                          const size_t* __restrict__ indsy,
                                          const double* __restrict__ weightsy,
                                          const double* __restrict__ dtratio,
                                          double* __restrict__ qK_result,
                                          double* __restrict__ qR_result,
                                          const double* __restrict__ QKv,
                                          const double* __restrict__ QRv,
                                          const double* __restrict__ dQKv,
                                          const double* __restrict__ dQRv,
                                          size_t len,
                                          size_t depth,
                                          size_t t1len,
                                          size_t prod) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= prod) return;

    size_t max_indx = (size_t)(posx[prod - 1] - 0.5);
    size_t indsx = max(min((size_t)(posx[j]), max_indx), (size_t)1);

    double inx = posx[j] - indsx;
    double inx2 = inx * inx;
    double inx3 = inx2 * inx;
    size_t inds = (indsx - 1) * len + indsy[j];

    const double* weights = weightsy + j * depth;

    double qK0 = 0.0, qK1 = 0.0, dqK1 = 0.0, dqK2 = 0.0;
    double qR0 = 0.0, qR1 = 0.0, dqR1 = 0.0, dqR2 = 0.0;

    for (size_t d = 0; d < depth; ++d) {
        size_t offset = inds + d;
        size_t offset1 = len + offset;
        size_t offset2 = 2 * len + offset;
        double w = weights[d];
        qK0  += w * QKv[offset];
        qK1  += w * QKv[offset1];
        dqK1 += w * dQKv[offset1];
        qR0  += w * QRv[offset];
        qR1  += w * QRv[offset1];
        dqR1 += w * dQRv[offset1];
        if (indsx >= t1len - 1) continue;
        dqK2 += w * dQKv[offset2];
        dqR2 += w * dQRv[offset2];
    }

    if (indsx < t1len - 1) {
        double denom = dtratio[indsx + 1];
        qK_result[j] = (1 - 3 * inx2 + 2 * inx3) * qK0 + (inx - 2 * inx2 + inx3) * dqK1 + (3 * inx2 - 2 * inx3) * qK1 + (-inx2 + inx3) * dqK2 / denom;
        qR_result[j] = (1 - 3 * inx2 + 2 * inx3) * qR0 + (inx - 2 * inx2 + inx3) * dqR1 + (3 * inx2 - 2 * inx3) * qR1 + (-inx2 + inx3) * dqR2 / denom;
    } else {
        qK_result[j] = (1 - inx2) * qK0 + inx2 * qK1 + (inx - inx2) * dqK1;
        qR_result[j] = (1 - inx2) * qR0 + inx2 * qR1 + (inx - inx2) * dqR1;
    }
}

// Compile-time specialized kernel for common depths (unrolled for speed)
template<int DEPTH>
__global__ __launch_bounds__(64, 1) void indexMatAllKernel_specialized(const double* __restrict__ posx,
                                          const size_t* __restrict__ indsy,
                                          const double* __restrict__ weightsy,
                                          const double* __restrict__ dtratio,
                                          double* __restrict__ qK_result,
                                          double* __restrict__ qR_result,
                                          const double* __restrict__ QKv,
                                          const double* __restrict__ QRv,
                                          const double* __restrict__ dQKv,
                                          const double* __restrict__ dQRv,
                                          size_t len,
                                          size_t t1len,
                                          size_t prod) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= prod) return;

    size_t max_indx = (size_t)(posx[prod - 1] - 0.5);
    size_t indsx = max(min((size_t)(posx[j]), max_indx), (size_t)1);

    double inx = posx[j] - indsx;
    double inx2 = inx * inx;
    double inx3 = inx2 * inx;
    size_t inds = (indsx - 1) * len + indsy[j];

    const double* weights = weightsy + j * DEPTH;

    double qK0 = 0.0, qK1 = 0.0, dqK1 = 0.0, dqK2 = 0.0;
    double qR0 = 0.0, qR1 = 0.0, dqR1 = 0.0, dqR2 = 0.0;

#pragma unroll
    for (int d = 0; d < DEPTH; ++d) {
        size_t offset = inds + d;
        size_t offset1 = len + offset;
        size_t offset2 = 2 * len + offset;
        double w = weights[d];
        qK0  += w * QKv[offset];
        qK1  += w * QKv[offset1];
        dqK1 += w * dQKv[offset1];
        qR0  += w * QRv[offset];
        qR1  += w * QRv[offset1];
        dqR1 += w * dQRv[offset1];
        if (indsx >= t1len - 1) continue;
        dqK2 += w * dQKv[offset2];
        dqR2 += w * dQRv[offset2];
    }

    if (indsx < t1len - 1) {
        double denom = dtratio[indsx + 1];
        qK_result[j] = (1 - 3 * inx2 + 2 * inx3) * qK0 + (inx - 2 * inx2 + inx3) * dqK1 + (3 * inx2 - 2 * inx3) * qK1 + (-inx2 + inx3) * dqK2 / denom;
        qR_result[j] = (1 - 3 * inx2 + 2 * inx3) * qR0 + (inx - 2 * inx2 + inx3) * dqR1 + (3 * inx2 - 2 * inx3) * qR1 + (-inx2 + inx3) * dqR2 / denom;
    } else {
        qK_result[j] = (1 - inx2) * qK0 + inx2 * qK1 + (inx - inx2) * dqK1;
        qR_result[j] = (1 - inx2) * qR0 + inx2 * qR1 + (inx - inx2) * dqR1;
    }
}

// Simple dispatcher selecting specialized vs dynamic kernel based on depth
class IndexMatAllOptimizer {
    static size_t cached_depth;
    using KernelFn = void(*)(const thrust::device_vector<double>&,
                             const thrust::device_vector<size_t>&,
                             const thrust::device_vector<double>&,
                             const thrust::device_vector<double>&,
                             thrust::device_vector<double>&,
                             thrust::device_vector<double>&,
                             const thrust::device_vector<double>&,
                             const thrust::device_vector<double>&,
                             const thrust::device_vector<double>&,
                             const thrust::device_vector<double>&,
                             size_t,
                             cudaStream_t);
    static KernelFn cached_kernel;

    template<int DEPTH>
    static void call_specialized(
        const thrust::device_vector<double>& posx,
        const thrust::device_vector<size_t>& indsy,
        const thrust::device_vector<double>& weightsy,
        const thrust::device_vector<double>& dtratio,
        thrust::device_vector<double>& qK_result,
        thrust::device_vector<double>& qR_result,
        const thrust::device_vector<double>& QKv,
        const thrust::device_vector<double>& QRv,
        const thrust::device_vector<double>& dQKv,
        const thrust::device_vector<double>& dQRv,
        size_t len,
        cudaStream_t stream) {
        size_t prod = indsy.size();
        size_t t1len = dtratio.size();
        int threads = 64;
        int blocks = (prod + threads - 1) / threads;
        indexMatAllKernel_specialized<DEPTH><<<blocks, threads, 0, stream>>>(
            thrust::raw_pointer_cast(posx.data()),
            thrust::raw_pointer_cast(indsy.data()),
            thrust::raw_pointer_cast(weightsy.data()),
            thrust::raw_pointer_cast(dtratio.data()),
            thrust::raw_pointer_cast(qK_result.data()),
            thrust::raw_pointer_cast(qR_result.data()),
            thrust::raw_pointer_cast(QKv.data()),
            thrust::raw_pointer_cast(QRv.data()),
            thrust::raw_pointer_cast(dQKv.data()),
            thrust::raw_pointer_cast(dQRv.data()),
            len, t1len, prod);
    }

    static void call_dynamic(
        const thrust::device_vector<double>& posx,
        const thrust::device_vector<size_t>& indsy,
        const thrust::device_vector<double>& weightsy,
        const thrust::device_vector<double>& dtratio,
        thrust::device_vector<double>& qK_result,
        thrust::device_vector<double>& qR_result,
        const thrust::device_vector<double>& QKv,
        const thrust::device_vector<double>& QRv,
        const thrust::device_vector<double>& dQKv,
        const thrust::device_vector<double>& dQRv,
        size_t len,
        cudaStream_t stream) {
        size_t prod = indsy.size();
        size_t depth = weightsy.size() / prod;
        size_t t1len = dtratio.size();
        int threads = 64;
        int blocks = (prod + threads - 1) / threads;
        indexMatAllKernel_dynamic<<<blocks, threads, 0, stream>>>(
            thrust::raw_pointer_cast(posx.data()),
            thrust::raw_pointer_cast(indsy.data()),
            thrust::raw_pointer_cast(weightsy.data()),
            thrust::raw_pointer_cast(dtratio.data()),
            thrust::raw_pointer_cast(qK_result.data()),
            thrust::raw_pointer_cast(qR_result.data()),
            thrust::raw_pointer_cast(QKv.data()),
            thrust::raw_pointer_cast(QRv.data()),
            thrust::raw_pointer_cast(dQKv.data()),
            thrust::raw_pointer_cast(dQRv.data()),
            len, depth, t1len, prod);
    }
public:
    static void setup(size_t depth) {
        if (cached_depth == depth) return;
        cached_depth = depth;
        switch(depth) {
            case 3:  cached_kernel = call_specialized<3>; break;
            case 5:  cached_kernel = call_specialized<5>; break;
            case 7:  cached_kernel = call_specialized<7>; break;
            case 9:  cached_kernel = call_specialized<9>; break;
            case 11: cached_kernel = call_specialized<11>; break;
            case 13: cached_kernel = call_specialized<13>; break;
            case 15: cached_kernel = call_specialized<15>; break;
            default: cached_kernel = call_dynamic; break;
        }
    }
    static void run(
        const thrust::device_vector<double>& posx,
        const thrust::device_vector<size_t>& indsy,
        const thrust::device_vector<double>& weightsy,
        const thrust::device_vector<double>& dtratio,
        thrust::device_vector<double>& qK_result,
        thrust::device_vector<double>& qR_result,
        const thrust::device_vector<double>& QKv,
        const thrust::device_vector<double>& QRv,
        const thrust::device_vector<double>& dQKv,
        const thrust::device_vector<double>& dQRv,
        size_t len,
        cudaStream_t stream) {
        cached_kernel(posx, indsy, weightsy, dtratio, qK_result, qR_result, QKv, QRv, dQKv, dQRv, len, stream);
    }
};

size_t IndexMatAllOptimizer::cached_depth = 0;
IndexMatAllOptimizer::KernelFn IndexMatAllOptimizer::cached_kernel = nullptr;

void indexMatAllGPU(const thrust::device_vector<double>& posx,
                    const thrust::device_vector<size_t>& indsy,
                    const thrust::device_vector<double>& weightsy,
                    const thrust::device_vector<double>& dtratio,
                    thrust::device_vector<double>& qK_result,
                    thrust::device_vector<double>& qR_result,
                    const thrust::device_vector<double>& QKv,
                    const thrust::device_vector<double>& QRv,
                    const thrust::device_vector<double>& dQKv,
                    const thrust::device_vector<double>& dQRv,
                    size_t len,
                    cudaStream_t stream) {
    const size_t depth = weightsy.size() / indsy.size();
    IndexMatAllOptimizer::setup(depth);
    IndexMatAllOptimizer::run(posx, indsy, weightsy, dtratio, qK_result, qR_result, QKv, QRv, dQKv, dQRv, len, stream);
}

// GPU interpolation functions
