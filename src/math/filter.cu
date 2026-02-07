#include "math/filter.hpp"
#include "core/globals.hpp"
#include "core/config.hpp"
#include "core/stream_pool.hpp"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <algorithm>
#include <iostream>

// External declarations for global variables
extern SimulationConfig config;
extern SimulationData* sim;

namespace {
__device__ __forceinline__ double smoothstep(double t) {
    return t * t * (3.0 - 2.0 * t);
}

__global__ void filter4_kernel_old(double* __restrict__ gK,
                                   double* __restrict__ gR,
                                   double* __restrict__ hK0,
                                   double* __restrict__ hR0,
                                   const double* __restrict__ theta,
                                   size_t len,
                                   double alpha,
                                   double dx_avg,
                                   int taper_len)
{
    extern __shared__ double s[];
    double* sK = s;
    double* sR = sK + (blockDim.x + 2);
    double* sHK = sR + (blockDim.x + 2);
    double* sHR = sHK + (blockDim.x + 2);

    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int sidx = tid + 1;
    const bool valid = i < len;

    if (valid) {
        sK[sidx] = gK[i];
        sR[sidx] = gR[i];
        sHK[sidx] = hK0[i];
        sHR[sidx] = hR0[i];
        if (tid == 0) {
            const size_t il = (i > 0) ? (i - 1) : i;
            sK[0] = gK[il];
            sR[0] = gR[il];
            sHK[0] = hK0[il];
            sHR[0] = hR0[il];
        }
        if (tid == blockDim.x - 1) {
            const size_t ir = (i + 1 < len) ? (i + 1) : i;
            sK[sidx + 1] = gK[ir];
            sR[sidx + 1] = gR[ir];
            sHK[sidx + 1] = hK0[ir];
            sHR[sidx + 1] = hR0[ir];
        }
    } else {
        sK[sidx] = 0.0;
        sR[sidx] = 0.0;
        sHK[sidx] = 0.0;
        sHR[sidx] = 0.0;
    }

    __syncthreads();

    if (!valid) return;
    if (i == 0 || i + 1 >= len) {
        gK[i] = sK[sidx];
        gR[i] = sR[sidx];
        hK0[i] = sHK[sidx];
        hR0[i] = sHR[sidx];
        return;
    }

    const double left = theta[i - 1];
    const double mid = theta[i];
    const double right = theta[i + 1];
    double dxm = mid - left;
    double dxp = right - mid;
    double denom = dxm + dxp;
    if (denom <= 0.0) denom = 1.0;

    double wL = dxp / denom;
    double wR = dxm / denom;

    int d = (int)(i < (len - 1 - i) ? i : (len - 1 - i));
    double taper = 1.0;
    if (taper_len > 0 && d < taper_len) {
        double t = static_cast<double>(d) / static_cast<double>(taper_len);
        taper = smoothstep(t);
    }

    double dx = 0.5 * (dxm + dxp);
    double wdx = (dx_avg > 0.0) ? (dx / dx_avg) : 1.0;
    if (wdx < 0.25) wdx = 0.25;
    if (wdx > 4.0) wdx = 4.0;

    double a = alpha * taper * wdx;
    if (a < 0.0) a = 0.0;
    if (a > 1.0) a = 1.0;

    double avgK = wL * sK[sidx - 1] + wR * sK[sidx + 1];
    double avgR = wL * sR[sidx - 1] + wR * sR[sidx + 1];
    double avgHK = wL * sHK[sidx - 1] + wR * sHK[sidx + 1];
    double avgHR = wL * sHR[sidx - 1] + wR * sHR[sidx + 1];

    gK[i] = sK[sidx] + a * (avgK - sK[sidx]);
    gR[i] = sR[sidx] + a * (avgR - sR[sidx]);
    hK0[i] = sHK[sidx] + a * (avgHK - sHK[sidx]);
    hR0[i] = sHR[sidx] + a * (avgHR - sHR[sidx]);
}

__global__ void filter4_lpf5_kernel(double* __restrict__ gK,
                                    double* __restrict__ gR,
                                    double* __restrict__ hK0,
                                    double* __restrict__ hR0,
                                    const double* __restrict__ theta,
                                    size_t len,
                                    double alpha,
                                    const double* __restrict__ apply_weight)
{
    if (apply_weight && (*apply_weight <= 0.0)) return;
    extern __shared__ double s[];
    double* sK = s;
    double* sR = sK + (blockDim.x + 4);
    double* sHK = sR + (blockDim.x + 4);
    double* sHR = sHK + (blockDim.x + 4);

    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int sidx = tid + 2;
    const bool valid = i < len;

    const bool hasK = (gK != nullptr);
    const bool hasR = (gR != nullptr);
    const bool hasHK = (hK0 != nullptr);
    const bool hasHR = (hR0 != nullptr);

    if (valid) {
        sK[sidx] = hasK ? gK[i] : 0.0;
        sR[sidx] = hasR ? gR[i] : 0.0;
        sHK[sidx] = hasHK ? hK0[i] : 0.0;
        sHR[sidx] = hasHR ? hR0[i] : 0.0;
    } else {
        sK[sidx] = 0.0;
        sR[sidx] = 0.0;
        sHK[sidx] = 0.0;
        sHR[sidx] = 0.0;
    }

    if (tid < 2) {
        size_t il = (i >= static_cast<size_t>(2 - tid)) ? (i - static_cast<size_t>(2 - tid)) : 0;
        if (il >= len) il = (len > 0 ? len - 1 : 0);
        sK[sidx - (2 - tid)] = hasK ? gK[il] : 0.0;
        sR[sidx - (2 - tid)] = hasR ? gR[il] : 0.0;
        sHK[sidx - (2 - tid)] = hasHK ? hK0[il] : 0.0;
        sHR[sidx - (2 - tid)] = hasHR ? hR0[il] : 0.0;
    }

    if (tid >= blockDim.x - 2) {
        int right_offset = tid - (blockDim.x - 2) + 1; // 1 or 2
        size_t ir = i + static_cast<size_t>(right_offset);
        if (ir >= len) ir = (len > 0 ? len - 1 : 0);
        sK[sidx + right_offset] = hasK ? gK[ir] : 0.0;
        sR[sidx + right_offset] = hasR ? gR[ir] : 0.0;
        sHK[sidx + right_offset] = hasHK ? hK0[ir] : 0.0;
        sHR[sidx + right_offset] = hasHR ? hR0[ir] : 0.0;
    }

    __syncthreads();

    if (!valid) return;
    if (i < 2 || i + 2 >= len) {
        if (hasK) gK[i] = sK[sidx];
        if (hasR) gR[i] = sR[sidx];
        if (hasHK) hK0[i] = sHK[sidx];
        if (hasHR) hR0[i] = sHR[sidx];
        return;
    }

    constexpr double c0 = -0.003571;  // Hamming-windowed sinc, cutoff=3 rad, width=5
    constexpr double c1 =  0.024358;
    constexpr double c2 =  0.958426;

    double dxm = theta[i] - theta[i - 1];
    double w = alpha * dxm * (apply_weight ? *apply_weight : 1.0);
    if (w < 0.0) w = 0.0;
    if (w > 1.0) w = 1.0;
    w *= 1e-2;

    double lpfK = c0 * (sK[sidx - 2] + sK[sidx + 2]) + c1 * (sK[sidx - 1] + sK[sidx + 1]) + c2 * sK[sidx];
    double lpfR = c0 * (sR[sidx - 2] + sR[sidx + 2]) + c1 * (sR[sidx - 1] + sR[sidx + 1]) + c2 * sR[sidx];
    double lpfHK = c0 * (sHK[sidx - 2] + sHK[sidx + 2]) + c1 * (sHK[sidx - 1] + sHK[sidx + 1]) + c2 * sHK[sidx];
    double lpfHR = c0 * (sHR[sidx - 2] + sHR[sidx + 2]) + c1 * (sHR[sidx - 1] + sHR[sidx + 1]) + c2 * sHR[sidx];

    if (hasK) gK[i] = sK[sidx] + w * (lpfK - sK[sidx]);
    if (hasR) gR[i] = sR[sidx] + w * (lpfR - sR[sidx]);
    if (hasHK) hK0[i] = sHK[sidx] + w * (lpfHK - sHK[sidx]);
    if (hasHR) hR0[i] = sHR[sidx] + w * (lpfHR - sHR[sidx]);
}

__global__ void filter_need_kernel(const double* __restrict__ x,
                                   const double* __restrict__ theta,
                                   size_t len,
                                   double* __restrict__ accum)
{
    constexpr double c0 = -0.003571;
    constexpr double c1 =  0.024358;
    constexpr double c2 =  0.958426;

    double local_num = 0.0;
    double local_den = 0.0;

    size_t i = blockIdx.x * blockDim.x + threadIdx.x + 3;
    const size_t stride = blockDim.x * gridDim.x;
    for (; i + 2 < len; i += stride) {
        double lpf_i = c0 * (x[i - 2] + x[i + 2])
                     + c1 * (x[i - 1] + x[i + 1])
                     + c2 * x[i];
        double dxm = theta[i] - theta[i - 1];
        double d_i = dxm * (lpf_i - x[i]);

        double lpf_im1 = c0 * (x[i - 3] + x[i + 1])
                       + c1 * (x[i - 2] + x[i])
                       + c2 * x[i - 1];
        double d_im1 = dxm * (lpf_im1 - x[i - 1]);

        local_num += fabs(d_i - d_im1);
        local_den += fabs(d_i + d_im1);
    }

    if (local_num != 0.0) atomicAdd(&accum[0], local_num);
    if (local_den != 0.0) atomicAdd(&accum[1], local_den);
}

__global__ void filter_need_finalize_kernel(const double* __restrict__ accum,
                                            double* __restrict__ apply_weight)
{
    if (!apply_weight) return;
    double threshold = 0.1;
    double den = accum[1];
    double num = accum[0];
    if (den <= 0.0) { *apply_weight = 0.0; return; }
    double ratio = num / den;
    *apply_weight = (ratio > threshold) ? (ratio - threshold) : 0.0;
}
} // namespace

void filter_old(thrust::device_ptr<double> gK,
                thrust::device_ptr<double> gR,
                thrust::device_ptr<double> hK0,
                thrust::device_ptr<double> hR0,
                size_t len,
                StreamPool& pool)
{
    if (len < 3) return;
    if (config.filter_strength <= 0.0) return;
    if (sim == nullptr) return;
    if (sim->d_theta.size() < len) return;

    double alpha = config.filter_strength;
    if (alpha < 0.0) return;
    if (alpha > 1.0) alpha = 1.0;

    double dx_avg = 1.0;
    if (sim->h_theta.size() >= len) {
        dx_avg = (sim->h_theta[len - 1] - sim->h_theta[0]) / static_cast<double>(len - 1);
        if (dx_avg <= 0.0) dx_avg = 1.0;
    }

    int taper_len = static_cast<int>(len / 20); // ~5% taper
    if (taper_len < 2) taper_len = 2;
    if (taper_len > 32) taper_len = 32;

    const double* theta_ptr = thrust::raw_pointer_cast(sim->d_theta.data());
    cudaStream_t stream = pool[0];

    int threads = 256;
    int blocks = static_cast<int>((len + threads - 1) / threads);
    size_t shmem = static_cast<size_t>(4 * (threads + 2)) * sizeof(double);
    filter4_kernel_old<<<blocks, threads, shmem, stream>>>(
        thrust::raw_pointer_cast(gK),
        thrust::raw_pointer_cast(gR),
        thrust::raw_pointer_cast(hK0),
        thrust::raw_pointer_cast(hR0),
        theta_ptr,
        len,
        alpha,
        dx_avg,
        taper_len);
}

void filter(thrust::device_ptr<double> gK,
            thrust::device_ptr<double> gR,
            thrust::device_ptr<double> hK0,
            thrust::device_ptr<double> hR0,
            size_t len,
            StreamPool& pool)
{
    if (len < 5) return;
    if (config.filter_strength <= 0.0) return;
    if (sim == nullptr) return;
    if (sim->d_theta.size() < len) return;
    if (hR0 == nullptr) return;

    double alpha = config.filter_strength;
    if (alpha < 0.0) return;

    const double* theta_ptr = thrust::raw_pointer_cast(sim->d_theta.data());
    cudaStream_t stream = pool[0];

    // Early-time guard: compute ratio on hR0 and only apply if weight > 0.
    static double* d_accum = nullptr;
    static double* d_apply = nullptr;
    if (d_accum == nullptr) {
        cudaMalloc(&d_accum, 2 * sizeof(double));
    }
    if (d_apply == nullptr) {
        cudaMalloc(&d_apply, sizeof(double));
    }
    cudaMemsetAsync(d_accum, 0, 2 * sizeof(double), stream);
    cudaMemsetAsync(d_apply, 0, sizeof(double), stream);

    int threads_need = 256;
    int blocks_need = static_cast<int>((len + threads_need - 1) / threads_need);
    if (blocks_need > 128) blocks_need = 128;
    filter_need_kernel<<<blocks_need, threads_need, 0, stream>>>(
        thrust::raw_pointer_cast(hR0),
        theta_ptr,
        len,
        d_accum);
    filter_need_finalize_kernel<<<1, 1, 0, stream>>>(d_accum, d_apply);

    double h_apply = 0.0;
    double h_accum[2] = {0.0, 0.0};
    cudaMemcpyAsync(&h_apply, d_apply, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_accum, d_accum, 2 * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (h_apply > 0.0) {
        std::cout << "filter applied (num=" << h_accum[0] << ", den=" << h_accum[1] << ", weight=" << h_apply << ")" << std::endl;
    }

    int threads = 256;
    int blocks = static_cast<int>((len + threads - 1) / threads);
    size_t shmem = static_cast<size_t>(4 * (threads + 4)) * sizeof(double);
    filter4_lpf5_kernel<<<blocks, threads, shmem, stream>>>(
        thrust::raw_pointer_cast(gK),
        thrust::raw_pointer_cast(gR),
        thrust::raw_pointer_cast(hK0),
        thrust::raw_pointer_cast(hR0),
        theta_ptr,
        len,
        alpha,
        d_apply);
}

void filter_dR(thrust::device_ptr<double> hR0,
                size_t len,
                StreamPool& pool)
{
    filter(nullptr, nullptr, nullptr, hR0, len, pool);
}

void filter_dRK(thrust::device_ptr<double> hK0,
                    thrust::device_ptr<double> hR0,
                    size_t len,
                    StreamPool& pool)
{
    filter(nullptr, nullptr, hK0, hR0, len, pool);
}
