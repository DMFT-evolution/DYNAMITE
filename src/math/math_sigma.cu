 #include "math/math_sigma.hpp"
 #include "core/device_utils.cuh"
 #include "math/math_ops.hpp"
 #include <vector>
 #include <thrust/device_vector.h>
 #include <thrust/transform.h>
 #include <thrust/iterator/zip_iterator.h>
 #include <thrust/execution_policy.h>
 #include <cassert>

// Minimal inline integer power (non-negative) used by device-side sigma helpers
__device__ __forceinline__ double fast_pow_int(double x, int n) {
    double r = 1.0;
    while (n > 0) {
        if (n & 1) r *= x;
        x *= x;
        n >>= 1;
    }
    return r;
}

// Device constants used by sigma computations
#include "core/device_constants.hpp"

// Device inline: evaluate SigmaK and SigmaR contributions

__device__ __forceinline__ void eval_sigmaK_sigmaR_fused(double qk, double qr, double& outK, double& outR) {
    const double q_p_minus_2  = fast_pow_int(qk, d_p - 2);
    const double q_p2_minus_2 = fast_pow_int(qk, d_p2 - 2);
    const double q_p_minus_1  = q_p_minus_2  * qk;
    const double q_p2_minus_1 = q_p2_minus_2 * qk;
    const double sK = d_lambda * d_p * q_p_minus_1 + (1.0 - d_lambda) * d_p2 * q_p2_minus_1;
    const double sR_base = d_lambda * d_p * (d_p - 1) * q_p_minus_2 + (1.0 - d_lambda) * d_p2 * (d_p2 - 1) * q_p2_minus_2;
    outK = sK;
    outR = sR_base * qr;
}

__global__ void computeSigmaKandRKernel(const double* __restrict__ qK,
                                        const double* __restrict__ qR,
                                        double* __restrict__ sigmaK,
                                        double* __restrict__ sigmaR,
                                        size_t len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int)len) return;
    double sK, sR;
    eval_sigmaK_sigmaR_fused(qK[i], qR[i], sK, sR);
    sigmaK[i] = sK;
    sigmaR[i] = sR;
}

// Host and device entry points for SigmaK/SigmaR
void SigmaKGPU(const thrust::device_vector<double>& qk, thrust::device_vector<double>& result, cudaStream_t stream) {
    assert(qk.size() == result.size());

    thrust::transform(
        thrust::cuda::par.on(stream),
        qk.begin(), qk.end(),
        result.begin(),
        [] __device__(double qk_val) {
            return DflambdaGPU(qk_val);
        }
    );
}

void SigmaRGPU(const thrust::device_vector<double>& qk, 
               const thrust::device_vector<double>& qr, 
               thrust::device_vector<double>& result,
               cudaStream_t stream) {
    assert(qk.size() == qr.size() && qk.size() == result.size());

    thrust::transform(
        thrust::cuda::par.on(stream),
        thrust::make_zip_iterator(thrust::make_tuple(qk.begin(), qr.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(qk.end(), qr.end())),
        result.begin(),
        [] __device__(thrust::tuple<double, double> qk_qr) {
            double qk = thrust::get<0>(qk_qr);
            double qr = thrust::get<1>(qk_qr);
            return DDflambdaGPU(qk) * qr;
        }
    );
}
