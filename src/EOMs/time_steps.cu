#include "EOMs/time_steps.hpp"
#include "core/globals.hpp"
#include "core/config.hpp"
#include "math/math_ops.hpp"
#include "core/vector_utils.hpp"
#include "convolution/convolution.hpp"
#include "core/compute_utils.hpp"
#include "math/math_sigma.hpp"
#include "io/io_utils.hpp"
#include "core/device_utils.cuh"
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <numeric>

using namespace std;

// External declaration for global config variable
extern SimulationConfig config;

// GPU-specific utility function for extracting last entries
thrust::device_vector<double> getLastLenEntriesGPU(const thrust::device_vector<double>& vec, size_t len) {
    if (len > vec.size()) {
        throw std::invalid_argument("len is greater than the size of the vector.");
    }
    return thrust::device_vector<double>(vec.end() - len, vec.end());
}

// Kernel implementations
__global__ void FusedQRKernel(
    const double* __restrict__ qR,
    const double* __restrict__ theta,
    const double* __restrict__ conv1,
    const double* __restrict__ conv2,
    const double* __restrict__ r,
    double* __restrict__ out,
    size_t len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double rEnd = r[len - 1];
    if (i < len) {
        out[i] = conv1[i] - qR[i] * rEnd + theta[i] * (qR[i] * r[i] - conv2[i]);
    }
}

__global__ void computeRstepResult(
    const double* __restrict__ temp0,
    const double* __restrict__ temp2,
    const double* __restrict__ temp3,
    const double* __restrict__ qK,
    double* __restrict__ result,
    double Gamma,
    double T0)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        result[0] = Gamma + temp2[0] + temp3[0] + temp0[0] * qK[0] / T0;
    }
}

__global__ void computeDrstepResult(
    const double* __restrict__ convA_sigmaR_qK,
    const double* __restrict__ convA_sigmaK_qR,
    const double* __restrict__ convA_dsigmaR_qK,
    const double* __restrict__ convA_dsigmaK_qR,
    const double* __restrict__ convA_sigmaR_dqK,
    const double* __restrict__ convA_sigmaK_dqR,
    const double* __restrict__ dsigmaK,
    const double* __restrict__ sigmaK,
    const double* __restrict__ QKv,
    const double* __restrict__ dQKv,
    double* __restrict__ result,
    double T0)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        result[0] = convA_sigmaR_qK[0] + convA_sigmaK_qR[0] + convA_dsigmaR_qK[0] + 
                    convA_dsigmaK_qR[0] + convA_sigmaR_dqK[0] + convA_sigmaK_dqR[0] +
                    (dsigmaK[0] * QKv[0] + sigmaK[0] * dQKv[0]) / T0;
    }
}

__global__ void computeDrstep2Result(
    const double* __restrict__ temp0,
    const double* __restrict__ temp2,
    const double* __restrict__ temp4,
    const double* __restrict__ temp5,
    const double* __restrict__ temp6,
    const double* __restrict__ temp7,
    const double* __restrict__ temp8,
    const double* __restrict__ temp9,
    const double* __restrict__ QKv,
    const double* __restrict__ dQKv,
    double* __restrict__ result,
    double T0)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        result[0] = temp4[0] + temp5[0] + temp6[0] + temp7[0] + temp8[0] + temp9[0] +
                    (temp2[0] * QKv[0] + temp0[0] * dQKv[0]) / T0;
    }
}

__global__ void computeCopy(
    const double* __restrict__ src,
    double* __restrict__ dest,
    size_t offset,
    size_t len,
    double factor)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        dest[offset + i] = factor * src[i];
    }
}

// Fused copy of qK, qR, dqK, dqR into contiguous history buffers.
// Scales derivative arrays by dt = t - t1grid[idx-1].
__global__ void append4Kernel(
    const double* __restrict__ qK,
    const double* __restrict__ qR,
    const double* __restrict__ dqK,
    const double* __restrict__ dqR,
    double* __restrict__ QKv,
    double* __restrict__ QRv,
    double* __restrict__ dQKv,
    double* __restrict__ dQRv,
    const double* __restrict__ t1grid,
    size_t idx,         // time index being appended
    size_t offset,      // element offset = previous size of QKv
    size_t len,
    double t)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    const double dt = t - t1grid[idx - 1];

    QKv[offset + i]  = qK[i];
    QRv[offset + i]  = qR[i];
    dQKv[offset + i] = dqK[i] * dt;
    dQRv[offset + i] = dqR[i] * dt;
}

// Update t1grid[idx], delta_t_ratio[idx], and drvec[idx] on device.
__global__ void updateTimeScalarsKernel(
    double* __restrict__ t1grid,
    double* __restrict__ delta_t_ratio,
    double* __restrict__ drvec,
    size_t idx,
    double t,
    double dr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const double tprev1 = t1grid[idx - 1];
        const double dt = t - tprev1;
        t1grid[idx] = t;
        if (idx > 1) {
            const double tprev2 = t1grid[idx - 2];
            delta_t_ratio[idx] = dt / (tprev1 - tprev2);
        } else {
            delta_t_ratio[idx] = 0.0;
        }
        drvec[idx] = dt * dr;
    }
}

// Fused replacement of the last slice (qK,qR,dqK,dqR) with scaling by dt using t1grid[idx-1].
__global__ void replace4Kernel(
    const double* __restrict__ qK,
    const double* __restrict__ qR,
    const double* __restrict__ dqK,
    const double* __restrict__ dqR,
    double* __restrict__ QKv,
    double* __restrict__ QRv,
    double* __restrict__ dQKv,
    double* __restrict__ dQRv,
    const double* __restrict__ t1grid,
    size_t idx,         // time index being replaced (typically last = size-1)
    size_t offset,      // element offset = idx * len
    size_t len,
    double t)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    const double dt = t - t1grid[idx - 1];

    QKv[offset + i]  = qK[i];
    QRv[offset + i]  = qR[i];
    dQKv[offset + i] = dqK[i] * dt;
    dQRv[offset + i] = dqR[i] * dt;
}

// GPU time-step functions (CPU versions are in time_steps.cpp)
void QRstepFused(const thrust::device_ptr<double>& qR,
                 const thrust::device_vector<double>& theta,
                 const thrust::device_vector<double>& conv1,
                 const thrust::device_vector<double>& conv2,
                 const thrust::device_vector<double>& r,
                 double* out,
                 cudaStream_t stream) {
    size_t len = theta.size();
    int threads = 64;
    int blocks = (len + threads - 1) / threads;

    FusedQRKernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(qR),
        thrust::raw_pointer_cast(theta.data()),
        thrust::raw_pointer_cast(conv1.data()),
        thrust::raw_pointer_cast(conv2.data()),
        thrust::raw_pointer_cast(r.data()),
        thrust::raw_pointer_cast(out), len
    );
}

thrust::device_vector<double> QKstepGPU(
    const thrust::device_vector<double>& QKv,
    const thrust::device_vector<double>& QRv,
    const thrust::device_vector<double>& QKB1int,
    const thrust::device_vector<double>& QKB2int,
    const thrust::device_vector<double>& QKA1int,
    const thrust::device_vector<double>& QRA1int,
    const thrust::device_vector<double>& QRA2int,
    const thrust::device_vector<double>& QRB1int,
    const thrust::device_vector<double>& SigmaRA1int,
    const thrust::device_vector<double>& SigmaRA2int,
    const thrust::device_vector<double>& SigmaKB1int,
    const thrust::device_vector<double>& SigmaKB2int,
    const thrust::device_vector<double>& SigmaKA1int,
    const thrust::device_vector<double>& SigmaRB1int,
    const thrust::device_vector<double>& integ,
    const thrust::device_vector<double>& theta,
    const thrust::device_vector<double>& t1grid,
    const thrust::device_vector<double>& rInt,
    double T0,
    double Gamma,
    StreamPool& pool) {
    
    size_t len = theta.size();
    size_t t1len = t1grid.size();
    int threads = 64;
    int blocks = (len + threads - 1) / threads;
    thrust::device_ptr<double> qK = get_slice_ptr(QKv,t1len-1,len);
    thrust::device_ptr<double> qR = get_slice_ptr(QRv,t1len-1,len);
    thrust::counting_iterator<size_t> idx_first(0);
    thrust::counting_iterator<size_t> idx_last = idx_first + len;

    thrust::for_each(thrust::cuda::par.on(pool[6]),
        idx_first, idx_last,
        [len, T0, Gamma, rInt_back = rInt.data() + (rInt.size() - 1), t1_back = t1grid.data() + (t1grid.size() - 1), ptr = QKB1int.data(), qk0 = QKv[(t1len - 1) * len], temp0 = sim->temp0.begin(), temp1 = sim->temp1.begin(), temp4 = sim->temp4.begin(), temp5 = sim->temp5.begin(), temp6 = sim->temp6.begin(), temp7 = sim->temp7.begin(), temp8 = sim->temp8.begin()] __device__ (size_t i) {
            temp0[i] = ptr[i * len];
            temp1[i] = DflambdaGPU(ptr[i * len]);
            if (i == 0) {
                temp4[0] = DflambdaGPU(qk0) / T0;
                temp5[0] = qk0 / T0;
                temp6[0] = -(*rInt_back);
                temp7[0] = 2.0 * Gamma;
                temp8[0] = *t1_back;
            }
        }
    );

    // Step 1: Run reductions
    ConvAGPU_Stream(SigmaRA1int, QKB1int, sim->convA1_1, sim->temp8, integ, theta, pool[0]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("energyGPU::ConvAGPU_Stream 1");
    ConvAGPU_Stream(SigmaKA1int, QRB1int, sim->convA2_1, sim->temp8, integ, theta, pool[1]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("energyGPU::ConvAGPU_Stream 2");
    ConvAGPU_Stream(QRA1int, SigmaKB1int, sim->convA1_2, sim->temp8, integ, theta, pool[2]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("energyGPU::ConvAGPU_Stream 3");
    ConvAGPU_Stream(QKA1int, SigmaRB1int, sim->convA2_2, sim->temp8, integ, theta, pool[3]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("energyGPU::ConvAGPU_Stream 4");
    ConvRGPU_Stream(SigmaRA2int, QKB2int, sim->convR_1, sim->temp8, integ, theta, pool[4]);
    ConvRGPU_Stream(QRA2int, SigmaKB2int, sim->convR_2, sim->temp8, integ, theta, pool[5]);

    // Synchronization with events
    std::vector<cudaEvent_t> events(7);
    for (int i = 0; i <= 6; ++i) {
        cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
        cudaEventRecord(events[i], pool[i]);
        cudaStreamWaitEvent(pool[7], events[i], 0);
        cudaStreamWaitEvent(pool[8], events[i], 0);
    }

    // Step 3: Fuse everything
    FusedUpdate(thrust::device_pointer_cast(sim->temp0.data()), qK, sim->temp2, thrust::raw_pointer_cast(sim->temp4.data()), thrust::raw_pointer_cast(sim->temp6.data()), nullptr, &sim->convR_1, &sim->convA1_1, &sim->convA2_1, nullptr, pool[7]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("energyGPU::FusedUpdate K");

    // Compute d2qK
    FusedUpdate(thrust::device_pointer_cast(sim->temp1.data()), qR, sim->temp3, thrust::raw_pointer_cast(sim->temp5.data()), thrust::raw_pointer_cast(sim->temp7.data()), &rInt, &sim->convR_2, &sim->convA1_2, &sim->convA2_2, qK, pool[8]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("energyGPU::FusedUpdate R");

    // Combine d1qK and d2qK
    computeProduct<<<blocks, threads, 0, pool[8]>>>(thrust::raw_pointer_cast(sim->temp3.data()),thrust::raw_pointer_cast(theta.data()),theta.size());
    if (config.debug) DMFE_CUDA_POSTLAUNCH("energyGPU::computeProduct");
    computeSum<<<blocks, threads, 0, pool[8]>>>(thrust::raw_pointer_cast(sim->temp2.data()),thrust::raw_pointer_cast(sim->temp3.data()),theta.size());
    if (config.debug) DMFE_CUDA_POSTLAUNCH("energyGPU::computeSum");

    return sim->temp2;
}

thrust::device_vector<double> QRstepGPU(
    const thrust::device_vector<double>& QRv,
    const thrust::device_vector<double>& rInt,
    const thrust::device_vector<double>& SigmaRA2int,
    const thrust::device_vector<double>& QRB2int,
    const thrust::device_vector<double>& QRA2int,
    const thrust::device_vector<double>& SigmaRB2int,
    const thrust::device_vector<double>& t1grid,
    const thrust::device_vector<double>& theta,
    StreamPool& pool) {
    
    size_t len = theta.size();
    size_t t1len = t1grid.size();
    sim->temp8[0] = t1grid.back();
    thrust::device_ptr<double> qR = get_slice_ptr(QRv, t1len - 1, len);

    ConvRGPU_Stream(SigmaRA2int, QRB2int, sim->convR_1, sim->temp8, sim->h_integ, theta, pool[0]);
    ConvRGPU_Stream(QRA2int, SigmaRB2int, sim->convR_2, sim->temp8, sim->h_integ, theta, pool[1]);

    QRstepFused(qR, theta, sim->convR_1, sim->convR_2, rInt, thrust::raw_pointer_cast(sim->temp2.data()));

    return sim->temp2;
}

void QKRstepGPU(
    const thrust::device_vector<double>& QKv,
    const thrust::device_vector<double>& QRv,
    const thrust::device_vector<double>& QKB1int,
    const thrust::device_vector<double>& QKB2int,
    const thrust::device_vector<double>& QKA1int,
    const thrust::device_vector<double>& QRA1int,
    const thrust::device_vector<double>& QRA2int,
    const thrust::device_vector<double>& QRB1int,
    const thrust::device_vector<double>& QRB2int,
    const thrust::device_vector<double>& SigmaRA1int,
    const thrust::device_vector<double>& SigmaRA2int,
    const thrust::device_vector<double>& SigmaKB1int,
    const thrust::device_vector<double>& SigmaKB2int,
    const thrust::device_vector<double>& SigmaKA1int,
    const thrust::device_vector<double>& SigmaRB1int,
    const thrust::device_vector<double>& SigmaRB2int,
    const thrust::device_vector<double>& integ,
    const thrust::device_vector<double>& theta,
    const thrust::device_vector<double>& t1grid,
    const thrust::device_vector<double>& rInt,
    thrust::device_vector<double>& outK,
    thrust::device_vector<double>& outR,
    double T0,
    double Gamma,
    int n,
    StreamPool& pool) {
    
    size_t len = theta.size();
    size_t t1len = t1grid.size();
    int threads = 64;
    int blocks = (len + threads - 1) / threads;
    thrust::device_ptr<double> qK = get_slice_ptr(QKv,t1len-1,len);
    thrust::device_ptr<double> qR = get_slice_ptr(QRv,t1len-1,len);
    thrust::counting_iterator<size_t> idx_first(0);
    thrust::counting_iterator<size_t> idx_last = idx_first + len;

    thrust::for_each(thrust::cuda::par.on(pool[0]),
        idx_first, idx_last,
        [len, T0, Gamma, rInt_back = rInt.data() + (rInt.size() - 1), t1_back = t1grid.data() + (t1len - 1), ptr = QKB1int.data(), qk0 = QKv[(t1len - 1) * len], temp0 = sim->temp0.begin(), temp1 = sim->temp1.begin(), temp4 = sim->temp4.begin(), temp5 = sim->temp5.begin(), temp6 = sim->temp6.begin(), temp7 = sim->temp7.begin(), temp8 = sim->temp8.begin()] __device__ (size_t i) {
            temp0[i] = ptr[i * len];
            temp1[i] = DflambdaGPU(ptr[i * len]);
            if (i == 0) {
                temp4[0] = DflambdaGPU(qk0) / T0;
                temp5[0] = qk0 / T0;
                temp6[0] = -(*rInt_back);
                temp7[0] = 2.0 * Gamma;
                temp8[0] = *t1_back;
            }
        }
    );

    cudaDeviceSynchronize();

    // Step 1: Run reductions
    ConvAGPU_Stream(SigmaRA1int, QKB1int, sim->convA1_1, sim->temp8, integ, theta, pool[1]);
    ConvAGPU_Stream(SigmaKA1int, QRB1int, sim->convA2_1, sim->temp8, integ, theta, pool[2]);
    ConvAGPU_Stream(QRA1int, SigmaKB1int, sim->convA1_2, sim->temp8, integ, theta, pool[3]);
    ConvAGPU_Stream(QKA1int, SigmaRB1int, sim->convA2_2, sim->temp8, integ, theta, pool[4]);
    ConvRGPU_Stream(SigmaRA2int, QKB2int, sim->convR_1, sim->temp8, integ, theta, pool[5]);
    ConvRGPU_Stream(QRA2int, SigmaKB2int, sim->convR_2, sim->temp8, integ, theta, pool[6]);
    ConvRGPU_Stream(SigmaRA2int, QRB2int, sim->convR_3, sim->temp8, integ, theta, pool[7]);
    ConvRGPU_Stream(QRA2int, SigmaRB2int, sim->convR_4, sim->temp8, integ, theta, pool[8]);

    cudaDeviceSynchronize();

    // Step 3: Fuse everything
    FusedUpdate(thrust::device_pointer_cast(sim->temp0.data()), qK, sim->temp2, thrust::raw_pointer_cast(sim->temp4.data()), thrust::raw_pointer_cast(sim->temp6.data()), nullptr, &sim->convR_1, &sim->convA1_1, &sim->convA2_1, nullptr, pool[9]);

    // Compute d2qK
    FusedUpdate(thrust::device_pointer_cast(sim->temp1.data()), qR, sim->temp3, thrust::raw_pointer_cast(sim->temp5.data()), thrust::raw_pointer_cast(sim->temp7.data()), &rInt, &sim->convR_2, &sim->convA1_2, &sim->convA2_2, qK, pool[10]);

    // Combine d1qK and d2qK
    computeProduct<<<blocks, threads, 0, pool[10]>>>(thrust::raw_pointer_cast(sim->temp3.data()),thrust::raw_pointer_cast(theta.data()),len);
    computeSum<<<blocks, threads, 0, pool[10]>>>(thrust::raw_pointer_cast(sim->temp2.data()),thrust::raw_pointer_cast(sim->temp3.data()),thrust::raw_pointer_cast(outK.data()) + n * len, len);

    QRstepFused(qR, theta, sim->convR_3, sim->convR_4, rInt, thrust::raw_pointer_cast(outR.data()) + n * len, pool[11]);
}

double rstepGPU(
    const thrust::device_vector<double>& QKv,
    const thrust::device_vector<double>& QRv,
    const thrust::device_vector<double>& t1grid,
    const thrust::device_vector<double>& integ,
    const thrust::device_vector<double>& theta,
    double Gamma,
    double T0,
    StreamPool& pool) {
    
    size_t len = theta.size();
    size_t t1len = t1grid.size();
    const double t = t1grid.back();

    thrust::device_ptr<double> qK = get_slice_ptr(QKv, t1len - 1, len);
    thrust::device_ptr<double> qR = get_slice_ptr(QRv, t1len - 1, len);

    int threads = 64;
    int blocks = (len + threads - 1) / threads;

    computeSigmaKandRKernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(qK),
        thrust::raw_pointer_cast(qR),
        thrust::raw_pointer_cast(sim->temp0.data()),
        thrust::raw_pointer_cast(sim->temp1.data()),
        len
    );

    ConvAGPU_Stream(sim->temp1, qK, sim->temp2, t, integ, theta, pool[1]);
    ConvAGPU_Stream(sim->temp0, qR, sim->temp3, t, integ, theta, pool[2]);

    // Fused final computation on GPU
    computeRstepResult<<<1, 1, 0>>>(
        thrust::raw_pointer_cast(sim->temp0.data()),
        thrust::raw_pointer_cast(sim->temp2.data()),
        thrust::raw_pointer_cast(sim->temp3.data()),
        thrust::raw_pointer_cast(qK),
        thrust::raw_pointer_cast(sim->temp4.data()),
        Gamma,
        T0
    );

    return sim->temp4[0];
}

double drstepGPU(
    const thrust::device_vector<double>& QKv,
    const thrust::device_vector<double>& QRv,
    const thrust::device_vector<double>& dQKv,
    const thrust::device_vector<double>& dQRv,
    const thrust::device_vector<double>& t1grid,
    const thrust::device_vector<double>& integ,
    const thrust::device_vector<double>& theta,
    double T0) {
    
    size_t len = QKv.size();
    const double t = t1grid.back();
    thrust::device_vector<double> sigmaK(len, 0.0);
    thrust::device_vector<double> sigmaR(len, 0.0);
    thrust::device_vector<double> dsigmaK(len, 0.0);
    thrust::device_vector<double> dsigmaR(len, 0.0);

    // Compute sigmaK
    thrust::transform(
        QKv.begin(), QKv.end(),
        sigmaK.begin(),
        [] __device__(double qk) { return DflambdaGPU(qk); }
    );

    // Compute sigmaR
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(QKv.begin(), QRv.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(QKv.end(), QRv.end())),
        sigmaR.begin(),
        [] __device__(thrust::tuple<double, double> qk_qr) {
            double qk = thrust::get<0>(qk_qr);
            double qr = thrust::get<1>(qk_qr);
            return DDflambdaGPU(qk) * qr;
        }
    );

    // Compute dsigmaK
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(QKv.begin(), dQKv.begin(), dQRv.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(QKv.end(), dQKv.end(), dQRv.end())),
        dsigmaK.begin(),
        [] __device__(thrust::tuple<double, double, double> qk_dqk_dqr) {
            double qk = thrust::get<0>(qk_dqk_dqr);
            double dqk = thrust::get<1>(qk_dqk_dqr);
            double dqr = thrust::get<2>(qk_dqk_dqr);
            return DDflambdaGPU(qk) * dqk + DflambdaGPU(qk) * dqr;
        }
    );

    // Compute dsigmaR
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(QKv.begin(), QRv.begin(), dQKv.begin(), dQRv.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(QKv.end(), QRv.end(), dQKv.end(), dQRv.end())),
        dsigmaR.begin(),
        [] __device__(thrust::tuple<double, double, double, double> qk_qr_dqk_dqr) {
            double qk = thrust::get<0>(qk_qr_dqk_dqr);
            double qr = thrust::get<1>(qk_qr_dqk_dqr);
            double dqk = thrust::get<2>(qk_qr_dqk_dqr);
            double dqr = thrust::get<3>(qk_qr_dqk_dqr);
            return DDDflambdaGPU(qk) * dqk * qr + DDflambdaGPU(qk) * dqr;
        }
    );

    // Compute convolution results
    thrust::device_vector<double> convA_sigmaR_qK = ConvAGPU(sigmaR, QKv, t, integ, theta);
    thrust::device_vector<double> convA_sigmaK_qR = ConvAGPU(sigmaK, QRv, t, integ, theta);
    thrust::device_vector<double> convA_dsigmaR_qK = ConvAGPU(dsigmaR, QKv, t, integ, theta);
    thrust::device_vector<double> convA_dsigmaK_qR = ConvAGPU(dsigmaK, QRv, t, integ, theta);
    thrust::device_vector<double> convA_sigmaR_dqK = ConvAGPU(sigmaR, dQKv, t, integ, theta);
    thrust::device_vector<double> convA_sigmaK_dqR = ConvAGPU(sigmaK, dQRv, t, integ, theta);

    // Compute final result
    computeDrstepResult<<<1, 1, 0>>>(
        thrust::raw_pointer_cast(convA_sigmaR_qK.data()),
        thrust::raw_pointer_cast(convA_sigmaK_qR.data()),
        thrust::raw_pointer_cast(convA_dsigmaR_qK.data()),
        thrust::raw_pointer_cast(convA_dsigmaK_qR.data()),
        thrust::raw_pointer_cast(convA_sigmaR_dqK.data()),
        thrust::raw_pointer_cast(convA_sigmaK_dqR.data()),
        thrust::raw_pointer_cast(dsigmaK.data()),
        thrust::raw_pointer_cast(sigmaK.data()),
        thrust::raw_pointer_cast(QKv.data()),
        thrust::raw_pointer_cast(dQKv.data()),
        thrust::raw_pointer_cast(sim->temp0.data()),
        T0
    );

    return sim->temp0[0];
}

double drstep2GPU(
    const thrust::device_ptr<double>& QKv,
    const thrust::device_ptr<double>& QRv,
    const thrust::device_ptr<double>& dQKv,
    const thrust::device_ptr<double>& dQRv,
    const double t,
    const double T0,
    StreamPool& pool) {
    
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(
        QKv, QRv, dQKv, dQRv,
        sim->temp0.begin(),
        sim->temp1.begin(),
        sim->temp2.begin(),
        sim->temp3.begin()
    ));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(
        QKv + config.len, QRv + config.len, dQKv + config.len, dQRv + config.len,
        sim->temp0.begin() + config.len,
        sim->temp1.begin() + config.len,
        sim->temp2.begin() + config.len,
        sim->temp3.begin() + config.len
    ));

    thrust::for_each(
        thrust::cuda::par.on(pool[0]),
        begin, end,
        [] __device__ (const thrust::tuple<
            double, double, double, double,
            double&, double&, double&, double&
        >& t) {
            double qk  = thrust::get<0>(t);
            double qr  = thrust::get<1>(t);
            double dqk = thrust::get<2>(t);
            double dqr = thrust::get<3>(t);

            double& sigmaK  = thrust::get<4>(t);
            double& sigmaR  = thrust::get<5>(t);
            double& dsigmaK = thrust::get<6>(t);
            double& dsigmaR = thrust::get<7>(t);

            sigmaK  = DflambdaGPU(qk);
            sigmaR  = DDflambdaGPU(qk) * qr;
            dsigmaK = DDflambdaGPU(qk) * dqk;
            dsigmaR = DDDflambdaGPU(qk) * dqk * qr + DDflambdaGPU(qk) * dqr;
        }
    );

    // Compute convolution results
    ConvAGPU_Stream(sim->temp1, QKv, sim->temp4, 1.0, sim->d_integ, sim->d_theta, pool[1]);
    ConvAGPU_Stream(sim->temp0, QRv, sim->temp5, 1.0, sim->d_integ, sim->d_theta, pool[2]);
    ConvAGPU_Stream(sim->temp3, QKv, sim->temp6, t, sim->d_integ, sim->d_theta, pool[3]);
    ConvAGPU_Stream(sim->temp2, QRv, sim->temp7, t, sim->d_integ, sim->d_theta, pool[4]);
    ConvAGPU_Stream(sim->temp1, dQKv, sim->temp8, t, sim->d_integ, sim->d_theta, pool[5]);
    ConvAGPU_Stream(sim->temp0, dQRv, sim->temp9, t, sim->d_integ, sim->d_theta, pool[6]);

    // Compute final result
    computeDrstep2Result<<<1, 1, 0>>>(
        thrust::raw_pointer_cast(sim->temp0.data()),
        thrust::raw_pointer_cast(sim->temp2.data()),
        thrust::raw_pointer_cast(sim->temp4.data()),
        thrust::raw_pointer_cast(sim->temp5.data()),
        thrust::raw_pointer_cast(sim->temp6.data()),
        thrust::raw_pointer_cast(sim->temp7.data()),
        thrust::raw_pointer_cast(sim->temp8.data()),
        thrust::raw_pointer_cast(sim->temp9.data()),
        thrust::raw_pointer_cast(QKv),
        thrust::raw_pointer_cast(dQKv),
        thrust::raw_pointer_cast(sim->temp1.data()),
        T0
    );

    return sim->temp1[0];
}

double energyGPU(
    const thrust::device_vector<double>& QKv,
    const thrust::device_vector<double>& QRv,
    const thrust::device_vector<double>& t1grid,
    const thrust::device_vector<double>& integ,
    const thrust::device_vector<double>& theta,
    double T0) {
    
    size_t len = theta.size();
    size_t t1len = t1grid.size();
    const double t = t1grid.back();

    thrust::device_ptr<double> qK = get_slice_ptr(QKv, t1len - 1, len);
    thrust::device_ptr<double> qR = get_slice_ptr(QRv, t1len- 1, len);

    int threads = 64;
    int blocks = (len + threads - 1) / threads;

    computeSigmaKandRKernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(qK),
        thrust::raw_pointer_cast(qR),
        thrust::raw_pointer_cast(sim->temp0.data()),
        thrust::raw_pointer_cast(sim->temp1.data()),
        len
    );

    // Compute convolution results
    ConvAGPU_Stream(sim->temp0, qR, sim->temp2, t, integ, theta);

    // Compute final result
    double result = - (sim->temp2[0] + sim->temp0[0] / T0);

    return move(result);
}

// GPU append/replace functions (CPU versions are in time_steps.cpp)
void appendGPU(thrust::device_vector<double>& dest,
                                        const thrust::device_vector<double>& src, double scale) {
    size_t required_size = dest.size() + src.size();

    if (dest.capacity() < required_size) {
        // Allocate a new vector with more capacity (e.g., double the required size)
        size_t new_capacity = std::max(required_size, 2 * dest.capacity());
        thrust::device_vector<double> tmp;
        tmp.reserve(new_capacity);            // Preallocate
        tmp.resize(dest.size());              // Set to current size
        thrust::copy(dest.begin(), dest.end(), tmp.begin());
        dest.swap(tmp);                       // Efficient ownership transfer
    }

    size_t insert_pos = dest.size();
    dest.resize(required_size);

    thrust::transform(
        src.begin(), src.end(),
        dest.begin() + insert_pos,
        [scale] __device__ (double val) {
            return val * scale;
        }
    );
}

void appendGPU_ptr(thrust::device_vector<double>& dest,
                                        const thrust::device_ptr<double>& src, double size, double scale, cudaStream_t stream) {
    size_t required_size = dest.size() + size;

    if (dest.capacity() < required_size) {
        // Allocate a new vector with more capacity
        dest.reserve(dest.size() + 1000 * size);
    }

    size_t insert_pos = dest.size();
    dest.resize(required_size);

    const int threads = 64;
    const int blocks = (config.len + threads - 1) / threads;

    computeCopy<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(src),
        thrust::raw_pointer_cast(dest.data()),
        insert_pos,
        size,
        scale
    );
}

void appendAllGPU(
    const thrust::device_vector<double>& qK,
    const thrust::device_vector<double>& qR,
    const thrust::device_vector<double>& dqK,
    const thrust::device_vector<double>& dqR,
    const double dr,
    const double t,
    StreamPool& pool)
{
    size_t length = qK.size();
    if (length != qR.size() || length != dqK.size() || length != dqR.size()) {
        throw invalid_argument("All input vectors must have the same size.");
    }

    // 1) update sim->d_t1grid and sim->d_delta_t_ratio
    sim->d_t1grid.push_back(t);
    size_t idx = sim->d_t1grid.size() - 1;
    double tdiff = sim->d_t1grid[idx] - sim->d_t1grid[idx - 1];
    if (idx > 1) {
        double prev = sim->d_t1grid[idx - 1] - sim->d_t1grid[idx - 2];
        sim->d_delta_t_ratio.push_back(tdiff / prev);
    }
    else {
        sim->d_delta_t_ratio.push_back(0.0);
    }

    appendGPU(sim->d_QKv,qK);
    appendGPU(sim->d_QRv,qR);
    appendGPU(sim->d_dQKv, dqK, tdiff);
    appendGPU(sim->d_dQRv, dqR, tdiff);

    // 2) finally update drvec and rvec
    sim->d_drvec.push_back(tdiff * dr);
    sim->d_rvec.push_back(rstepGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, config.Gamma, config.T0, pool));
}

void appendAllGPU_ptr(
    const thrust::device_ptr<double>& qK,
    const thrust::device_ptr<double>& qR,
    const thrust::device_ptr<double>& dqK,
    const thrust::device_ptr<double>& dqR,
    const double dr,
    const double t,
    const size_t len,
    StreamPool& pool)
{
    // Current time index before append
    const size_t idx = sim->d_t1grid.size();

    // Keep existing resize cadence: grow by exactly one time-step when needed
    if (sim->d_t1grid.capacity() < idx + 1) {
        sim->d_t1grid.reserve(idx + 1000);
        sim->d_delta_t_ratio.reserve(idx + 1000);
    }

    // Ensure scalar timelines have space for the new entry
    sim->d_t1grid.resize(idx + 1);
    sim->d_delta_t_ratio.resize(idx + 1);
    sim->d_drvec.resize(idx + 1);
    sim->d_rvec.resize(idx + 1);

    // History buffers: compute offset and resize by +len (no change in frequency)
    const size_t offset = sim->d_QKv.size();
    if (sim->d_QKv.capacity() < offset + len) sim->d_QKv.reserve(offset + 1000 * len);
    if (sim->d_QRv.capacity() < offset + len) sim->d_QRv.reserve(offset + 1000 * len);
    if (sim->d_dQKv.capacity() < offset + len) sim->d_dQKv.reserve(offset + 1000 * len);
    if (sim->d_dQRv.capacity() < offset + len) sim->d_dQRv.reserve(offset + 1000 * len);
    sim->d_QKv.resize(offset + len);
    sim->d_QRv.resize(offset + len);
    sim->d_dQKv.resize(offset + len);
    sim->d_dQRv.resize(offset + len);

    // Launch fused history append (1 kernel instead of 4)
    const int threads = 64;
    const int blocks  = static_cast<int>((len + threads - 1) / threads);
    append4Kernel<<<blocks, threads, 0, pool[0]>>>(
        thrust::raw_pointer_cast(qK),
        thrust::raw_pointer_cast(qR),
        thrust::raw_pointer_cast(dqK),
        thrust::raw_pointer_cast(dqR),
        thrust::raw_pointer_cast(sim->d_QKv.data()),
        thrust::raw_pointer_cast(sim->d_QRv.data()),
        thrust::raw_pointer_cast(sim->d_dQKv.data()),
        thrust::raw_pointer_cast(sim->d_dQRv.data()),
        thrust::raw_pointer_cast(sim->d_t1grid.data()),
        idx,
        offset,
        len,
        t);

    // Update time scalars (t1grid[idx], delta[idx], drvec[idx]) on device
    updateTimeScalarsKernel<<<1, 1, 0, pool[0]>>>(
        thrust::raw_pointer_cast(sim->d_t1grid.data()),
        thrust::raw_pointer_cast(sim->d_delta_t_ratio.data()),
        thrust::raw_pointer_cast(sim->d_drvec.data()),
        idx,
        t,
        dr);

    // Compute r(t) and store it in d_rvec[idx]
    const double r_now = rstepGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, config.Gamma, config.T0, pool);
    sim->d_rvec[idx] = r_now;
}

void replaceAllGPU(
    const thrust::device_vector<double>& qK,
    const thrust::device_vector<double>& qR,
    const thrust::device_vector<double>& dqK,
    const thrust::device_vector<double>& dqR,
    const double dr,
    const double t,
    StreamPool& pool)
{   
    // Replace the existing values in the vectors with the new values
    size_t replaceLength = qK.size();
    size_t length = sim->d_QKv.size() - replaceLength;
    if (replaceLength != qR.size() || replaceLength != dqK.size() || replaceLength != dqR.size()) {
        throw invalid_argument("All input vectors must have the same size.");
    }
    {
        sim->d_t1grid.back() = t;
        double tdiff = (sim->d_t1grid[sim->d_t1grid.size() - 1] - sim->d_t1grid[sim->d_t1grid.size() - 2]);

        if (sim->d_t1grid.size() > 2) {
            sim->d_delta_t_ratio.back() = tdiff /
                (sim->d_t1grid[sim->d_t1grid.size() - 2] - sim->d_t1grid[sim->d_t1grid.size() - 3]);
        }
        else {
            sim->d_delta_t_ratio.back() = 0.0;
        }

        thrust::copy(qK.begin(), qK.end(), sim->d_QKv.begin() + length);
        thrust::copy(qR.begin(), qR.end(), sim->d_QRv.begin() + length);
        thrust::transform(dqK.begin(), dqK.end(), sim->d_dQKv.begin() + length, [tdiff] __device__ (double x) { return tdiff * x; });
        thrust::transform(dqR.begin(), dqR.end(), sim->d_dQRv.begin() + length, [tdiff] __device__ (double x) { return tdiff * x; });

        sim->d_drvec.back() = tdiff * dr;
        sim->d_rvec.back() = rstepGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, config.Gamma, config.T0, pool);
    }
}

void replaceAllGPU_ptr(
    const thrust::device_ptr<double>& qK,
    const thrust::device_ptr<double>& qR,
    const thrust::device_ptr<double>& dqK,
    const thrust::device_ptr<double>& dqR,
    const double dr,
    const double t,
    const size_t len,
    StreamPool& pool)
{   
    // Replace the existing values in the vectors with the new values (last slice)
    const size_t gridSize = sim->d_t1grid.size();
    const size_t idx = gridSize - 1;
    const size_t offset = sim->d_QKv.size() - len;

    const int threads = 64;
    const int blocks  = static_cast<int>((len + threads - 1) / threads);

    // Fused replace of 4 arrays with dt scaling computed on device
    replace4Kernel<<<blocks, threads, 0, pool[0]>>>(
        thrust::raw_pointer_cast(qK),
        thrust::raw_pointer_cast(qR),
        thrust::raw_pointer_cast(dqK),
        thrust::raw_pointer_cast(dqR),
        thrust::raw_pointer_cast(sim->d_QKv.data()),
        thrust::raw_pointer_cast(sim->d_QRv.data()),
        thrust::raw_pointer_cast(sim->d_dQKv.data()),
        thrust::raw_pointer_cast(sim->d_dQRv.data()),
        thrust::raw_pointer_cast(sim->d_t1grid.data()),
        idx,
        offset,
        len,
        t);

    // Update time scalars for the last entry
    updateTimeScalarsKernel<<<1, 1, 0, pool[0]>>>(
        thrust::raw_pointer_cast(sim->d_t1grid.data()),
        thrust::raw_pointer_cast(sim->d_delta_t_ratio.data()),
        thrust::raw_pointer_cast(sim->d_drvec.data()),
        idx,
        t,
        dr);

    // Update r(t) at the last position
    const double r_now = rstepGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, config.Gamma, config.T0, pool);
    sim->d_rvec.back() = r_now;
}
