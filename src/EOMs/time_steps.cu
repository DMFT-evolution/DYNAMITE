#include "EOMs/time_steps_cuda.cuh"
#include "core/globals.hpp"
#include "simulation/device_simulation_data.hpp"
#include "core/config.hpp"
#include "math/math_ops.hpp"
#include "core/vector_utils.cuh"
#include "convolution/convolution.cuh"
#include "core/compute_utils.hpp"
#include "math/math_sigma.cuh"
#include "io/io_utils.hpp"
#include "core/device_utils.cuh"
#include "core/gpu_events.hpp"
#include "core/t1_cache.hpp"
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
dmfe::device_vector<double> getLastLenEntriesGPU(const dmfe::device_vector<double>& vec, size_t len) {
    if (len > vec.size()) {
        throw std::invalid_argument("len is greater than the size of the vector.");
    }
    return dmfe::device_vector<double>(vec.end() - len, vec.end());
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
    // [PERF] t1grid is read (idx-1, idx-2) and written (idx) by this kernel now,
    // so it is a single non-const pointer: two __restrict__ pointers to the same
    // array, one of them written, would be undefined behaviour.
    double* __restrict__ t1grid,
    double* __restrict__ delta_t_ratio,
    double* __restrict__ drvec,
    size_t idx,         // time index being appended
    size_t offset,      // element offset = previous size of QKv
    size_t len,
    double t,
    double dr)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    const double dt = t - t1grid[idx - 1];

    QKv[offset + i]  = qK[i];
    QRv[offset + i]  = qR[i];
    dQKv[offset + i] = dqK[i] * dt;
    dQRv[offset + i] = dqR[i] * dt;

    // [PERF] The scalar update that used to be a separate <<<1,1>>> launch
    // (updateTimeScalarsKernel, on another stream) is done here by one thread.  It
    // reads t1grid[idx-1] and t1grid[idx-2], which this kernel does not write, and
    // writes t1grid[idx], delta_t_ratio[idx] and drvec[idx], which nothing here
    // reads; `dt` is the same expression as above.  Same values, one launch fewer,
    // and the two kernels no longer race to store the same t1grid[idx].
    if (i == 0) {
        t1grid[idx] = t;
        if (idx > 1) {
            delta_t_ratio[idx] = dt / (t1grid[idx - 1] - t1grid[idx - 2]);
        } else {
            delta_t_ratio[idx] = 0.0;
        }
        drvec[idx] = dt * dr;
    }
}

// [PERF] updateTimeScalarsKernel used to live here: a <<<1,1>>> launch, on a
// second stream, that wrote t1grid[idx], delta_t_ratio[idx] and drvec[idx].  It is
// now the `i == 0` branch of append4Kernel / replace4Kernel above, which already
// had `dt` in hand -- one launch fewer per stage, and the two kernels no longer
// both store t1grid[idx].

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
    // [PERF] t1grid is read (idx-1, idx-2) and written (idx) by this kernel now,
    // so it is a single non-const pointer: two __restrict__ pointers to the same
    // array, one of them written, would be undefined behaviour.
    double* __restrict__ t1grid,
    double* __restrict__ delta_t_ratio,
    double* __restrict__ drvec,
    size_t idx,         // time index being replaced (typically last = size-1)
    size_t offset,      // element offset = idx * len
    size_t len,
    double t,
    double dr)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    const double dt = t - t1grid[idx - 1];

    QKv[offset + i]  = qK[i];
    QRv[offset + i]  = qR[i];
    dQKv[offset + i] = dqK[i] * dt;
    dQRv[offset + i] = dqR[i] * dt;

    // [PERF] The scalar update that used to be a separate <<<1,1>>> launch
    // (updateTimeScalarsKernel, on another stream) is done here by one thread.  It
    // reads t1grid[idx-1] and t1grid[idx-2], which this kernel does not write, and
    // writes t1grid[idx], delta_t_ratio[idx] and drvec[idx], which nothing here
    // reads; `dt` is the same expression as above.  Same values, one launch fewer,
    // and the two kernels no longer race to store the same t1grid[idx].
    if (i == 0) {
        t1grid[idx] = t;
        if (idx > 1) {
            delta_t_ratio[idx] = dt / (t1grid[idx - 1] - t1grid[idx - 2]);
        } else {
            delta_t_ratio[idx] = 0.0;
        }
        drvec[idx] = dt * dr;
    }
}

// GPU time-step functions (CPU versions are in time_steps.cpp)
void QRstepFused(const thrust::device_ptr<double>& qR,
                    const dmfe::device_vector<double>& theta,
                    const dmfe::device_vector<double>& conv1,
                    const dmfe::device_vector<double>& conv2,
                    const dmfe::device_vector<double>& r,
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

dmfe::device_vector<double> QKstepGPU(
    const dmfe::device_vector<double>& QKv,
    const dmfe::device_vector<double>& QRv,
    const dmfe::device_vector<double>& QKB1int,
    const dmfe::device_vector<double>& QKB2int,
    const dmfe::device_vector<double>& QKA1int,
    const dmfe::device_vector<double>& QRA1int,
    const dmfe::device_vector<double>& QRA2int,
    const dmfe::device_vector<double>& QRB1int,
    const dmfe::device_vector<double>& SigmaRA1int,
    const dmfe::device_vector<double>& SigmaRA2int,
    const dmfe::device_vector<double>& SigmaKB1int,
    const dmfe::device_vector<double>& SigmaKB2int,
    const dmfe::device_vector<double>& SigmaKA1int,
    const dmfe::device_vector<double>& SigmaRB1int,
    const dmfe::device_vector<double>& integ,
    const dmfe::device_vector<double>& theta,
    const dmfe::device_vector<double>& t1grid,
    const dmfe::device_vector<double>& rInt,
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

    thrust::for_each(thrust::cuda::par_nosync.on(pool[6]),
        idx_first, idx_last,
        [len, T0, Gamma, rInt_back = rInt.data() + (rInt.size() - 1), t1_back = t1grid.data() + (t1grid.size() - 1), ptr = QKB1int.data(), qk0 = QKv[(t1len - 1) * len], temp0 = sim->device->temp0.begin(), temp1 = sim->device->temp1.begin(), temp4 = sim->device->temp4.begin(), temp5 = sim->device->temp5.begin(), temp6 = sim->device->temp6.begin(), temp7 = sim->device->temp7.begin(), temp8 = sim->device->temp8.begin()] __device__ (size_t i) {
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
    ConvAGPU_Stream(SigmaRA1int, QKB1int, sim->device->convA1_1, sim->device->temp8, integ, theta, pool[0]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("energyGPU::ConvAGPU_Stream 1");
    ConvAGPU_Stream(SigmaKA1int, QRB1int, sim->device->convA2_1, sim->device->temp8, integ, theta, pool[1]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("energyGPU::ConvAGPU_Stream 2");
    ConvAGPU_Stream(QRA1int, SigmaKB1int, sim->device->convA1_2, sim->device->temp8, integ, theta, pool[2]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("energyGPU::ConvAGPU_Stream 3");
    ConvAGPU_Stream(QKA1int, SigmaRB1int, sim->device->convA2_2, sim->device->temp8, integ, theta, pool[3]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("energyGPU::ConvAGPU_Stream 4");
    ConvRGPU_Stream(SigmaRA2int, QKB2int, sim->device->convR_1, sim->device->temp8, integ, theta, pool[4]);
    ConvRGPU_Stream(QRA2int, SigmaKB2int, sim->device->convR_2, sim->device->temp8, integ, theta, pool[5]);

    // Synchronization with events
    std::vector<cudaEvent_t> events(7);
    for (int i = 0; i <= 6; ++i) {
        cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
        cudaEventRecord(events[i], pool[i]);
        cudaStreamWaitEvent(pool[7], events[i], 0);
        cudaStreamWaitEvent(pool[8], events[i], 0);
    }

    // Step 3: Fuse everything
    FusedUpdate(thrust::device_pointer_cast(sim->device->temp0.data()), qK, sim->device->temp2, thrust::raw_pointer_cast(sim->device->temp4.data()), thrust::raw_pointer_cast(sim->device->temp6.data()), nullptr, &sim->device->convR_1, &sim->device->convA1_1, &sim->device->convA2_1, nullptr, pool[7]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("energyGPU::FusedUpdate K");

    // Compute d2qK
    FusedUpdate(thrust::device_pointer_cast(sim->device->temp1.data()), qR, sim->device->temp3, thrust::raw_pointer_cast(sim->device->temp5.data()), thrust::raw_pointer_cast(sim->device->temp7.data()), &rInt, &sim->device->convR_2, &sim->device->convA1_2, &sim->device->convA2_2, qK, pool[8]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("energyGPU::FusedUpdate R");

    // Combine d1qK and d2qK
    computeProduct<<<blocks, threads, 0, pool[8]>>>(thrust::raw_pointer_cast(sim->device->temp3.data()),thrust::raw_pointer_cast(theta.data()),theta.size());
    if (config.debug) DMFE_CUDA_POSTLAUNCH("energyGPU::computeProduct");
    computeSum<<<blocks, threads, 0, pool[8]>>>(thrust::raw_pointer_cast(sim->device->temp2.data()),thrust::raw_pointer_cast(sim->device->temp3.data()),theta.size());
    if (config.debug) DMFE_CUDA_POSTLAUNCH("energyGPU::computeSum");

    return sim->device->temp2;
}

dmfe::device_vector<double> QRstepGPU(
    const dmfe::device_vector<double>& QRv,
    const dmfe::device_vector<double>& rInt,
    const dmfe::device_vector<double>& SigmaRA2int,
    const dmfe::device_vector<double>& QRB2int,
    const dmfe::device_vector<double>& QRA2int,
    const dmfe::device_vector<double>& SigmaRB2int,
    const dmfe::device_vector<double>& t1grid,
    const dmfe::device_vector<double>& theta,
    StreamPool& pool) {
    
    size_t len = theta.size();
    size_t t1len = t1grid.size();
    sim->device->temp8[0] = t1grid.back();
    thrust::device_ptr<double> qR = get_slice_ptr(QRv, t1len - 1, len);

    ConvRGPU_Stream(SigmaRA2int, QRB2int, sim->device->convR_1, sim->device->temp8, sim->host->integ, theta, pool[0]);
    ConvRGPU_Stream(QRA2int, SigmaRB2int, sim->device->convR_2, sim->device->temp8, sim->host->integ, theta, pool[1]);

    QRstepFused(qR, theta, sim->device->convR_1, sim->device->convR_2, rInt, thrust::raw_pointer_cast(sim->device->temp2.data()));

    return sim->device->temp2;
}

void QKRstepGPU(
    const dmfe::device_vector<double>& QKv,
    const dmfe::device_vector<double>& QRv,
    const dmfe::device_vector<double>& QKB1int,
    const dmfe::device_vector<double>& QKB2int,
    const dmfe::device_vector<double>& QKA1int,
    const dmfe::device_vector<double>& QRA1int,
    const dmfe::device_vector<double>& QRA2int,
    const dmfe::device_vector<double>& QRB1int,
    const dmfe::device_vector<double>& QRB2int,
    const dmfe::device_vector<double>& SigmaRA1int,
    const dmfe::device_vector<double>& SigmaRA2int,
    const dmfe::device_vector<double>& SigmaKB1int,
    const dmfe::device_vector<double>& SigmaKB2int,
    const dmfe::device_vector<double>& SigmaKA1int,
    const dmfe::device_vector<double>& SigmaRB1int,
    const dmfe::device_vector<double>& SigmaRB2int,
    const dmfe::device_vector<double>& integ,
    const dmfe::device_vector<double>& theta,
    const dmfe::device_vector<double>& t1grid,
    const dmfe::device_vector<double>& rInt,
    dmfe::device_vector<double>& outK,
    dmfe::device_vector<double>& outR,
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
    // [PERF] These events used to be created (and never destroyed) on every call,
    // i.e. ~2 per Runge-Kutta stage.  They are stateless handles, so one static
    // instance is enough; this removes two API calls and a leak per stage.
    static cudaEvent_t ready   = nullptr;
    static cudaEvent_t convDone = nullptr;
    if (!ready) {
        cudaEventCreateWithFlags(&ready,    cudaEventDisableTiming);
        cudaEventCreateWithFlags(&convDone, cudaEventDisableTiming);
    }

    thrust::for_each(thrust::cuda::par_nosync.on(pool[0]),
        idx_first, idx_last,
        // [PERF] `qk0 = QKv[(t1len - 1) * len]` used to be captured by value: a
        // thrust::device_reference conversion, i.e. a one-element blocking
        // device->host copy -- and therefore an implicit device-wide barrier --
        // executed once per Runge-Kutta stage.  The element is now read on the
        // device instead.  It is the same double at the same address, used in the
        // same two expressions, so temp4[0] and temp5[0] are bit for bit what they
        // were.
        [len, T0, Gamma, rInt_back = rInt.data() + (rInt.size() - 1), t1_back = t1grid.data() + (t1len - 1), ptr = QKB1int.data(), qk0_ptr = thrust::raw_pointer_cast(QKv.data()) + (t1len - 1) * len, temp0 = sim->device->temp0.begin(), temp1 = sim->device->temp1.begin(), temp4 = sim->device->temp4.begin(), temp5 = sim->device->temp5.begin(), temp6 = sim->device->temp6.begin(), temp7 = sim->device->temp7.begin(), temp8 = sim->device->temp8.begin()] __device__ (size_t i) {
            temp0[i] = ptr[i * len];
            temp1[i] = DflambdaGPU(ptr[i * len]);
            if (i == 0) {
                const double qk0 = *qk0_ptr;
                temp4[0] = DflambdaGPU(qk0) / T0;
                temp5[0] = qk0 / T0;
                temp6[0] = -(*rInt_back);
                temp7[0] = 2.0 * Gamma;
                temp8[0] = *t1_back;
            }
        }
    );

    // [PERF] cudaDeviceSynchronize() removed.  The batched convolution below reads
    // temp8, written by the thrust::for_each above; issuing it on the SAME stream
    // (pool[0]) makes that dependency implicit, so no host-side barrier is needed.
    // Ordering, not arithmetic, changes: every kernel still reads exactly the same
    // values it read before.

    // Step 1: Run reductions
    // [PERF] The eight reductions used to be eight separate launches of
    // (len x 64) threads on eight streams.  They are now a single launch with a
    // (len x 8) grid: the same blocks, the same blockDim and the same per-block
    // reduction, so every value is bit-for-bit what it was, but seven launches
    // and their stream bookkeeping disappear and the occupancy grows eightfold.
    {
        ConvBatchDesc batch;
        auto P = [](const dmfe::device_vector<double>& v) { return thrust::raw_pointer_cast(v.data()); };
        auto Q = [](dmfe::device_vector<double>& v) { return thrust::raw_pointer_cast(v.data()); };
        batch.add(ConvKind::A, P(SigmaRA1int), P(QKB1int),     Q(sim->device->convA1_1));
        batch.add(ConvKind::A, P(SigmaKA1int), P(QRB1int),     Q(sim->device->convA2_1));
        batch.add(ConvKind::A, P(QRA1int),     P(SigmaKB1int), Q(sim->device->convA1_2));
        batch.add(ConvKind::A, P(QKA1int),     P(SigmaRB1int), Q(sim->device->convA2_2));
        batch.add(ConvKind::R, P(SigmaRA2int), P(QKB2int),     Q(sim->device->convR_1));
        batch.add(ConvKind::R, P(QRA2int),     P(SigmaKB2int), Q(sim->device->convR_2));
        batch.add(ConvKind::R, P(SigmaRA2int), P(QRB2int),     Q(sim->device->convR_3));
        batch.add(ConvKind::R, P(QRA2int),     P(SigmaRB2int), Q(sim->device->convR_4));
        ConvBatchedGPU(batch, integ, theta, sim->device->temp8, len, len, pool[0]);
    }

    // [PERF] cudaDeviceSynchronize() replaced by a single event.  The three
    // consumers below run on pool[9], pool[10] and pool[11] and read the outputs of
    // the batched convolution, which is issued on pool[0]; one recorded event and
    // three waits express exactly that dependency without blocking the host.
    cudaEventRecord(convDone, pool[0]);
    cudaStreamWaitEvent(pool[9],  convDone, 0);
    cudaStreamWaitEvent(pool[10], convDone, 0);
    cudaStreamWaitEvent(pool[11], convDone, 0);

    // Step 3: Fuse everything
    FusedUpdate(thrust::device_pointer_cast(sim->device->temp0.data()), qK, sim->device->temp2, thrust::raw_pointer_cast(sim->device->temp4.data()), thrust::raw_pointer_cast(sim->device->temp6.data()), nullptr, &sim->device->convR_1, &sim->device->convA1_1, &sim->device->convA2_1, nullptr, pool[9]);

    // Compute d2qK
    // [PERF] the `theta` factor that used to be applied by a separate
    // computeProduct launch right after this one is now the last operation of the
    // fused kernel itself (see FusedUpdateKernel in core/device_utils.cu).
    FusedUpdate(thrust::device_pointer_cast(sim->device->temp1.data()), qR, sim->device->temp3, thrust::raw_pointer_cast(sim->device->temp5.data()), thrust::raw_pointer_cast(sim->device->temp7.data()), &rInt, &sim->device->convR_2, &sim->device->convA1_2, &sim->device->convA2_2, qK, pool[10], &theta);

    // Combine d1qK and d2qK
    cudaEventRecord(ready, pool[9]);
    cudaStreamWaitEvent(pool[10], ready, 0);
    computeSum<<<blocks, threads, 0, pool[10]>>>(thrust::raw_pointer_cast(sim->device->temp2.data()),thrust::raw_pointer_cast(sim->device->temp3.data()),thrust::raw_pointer_cast(outK.data()) + n * len, len);

    QRstepFused(qR, theta, sim->device->convR_3, sim->device->convR_4, rInt, thrust::raw_pointer_cast(outR.data()) + n * len, pool[11]);
}

// [PERF] Same computation as rstepGPU() below, but the scalar r(t) is stored
// wherever the caller wants it instead of being copied to the host and copied
// straight back into a device array.  computeRstepResult evaluates exactly the
// same expression from exactly the same inputs; only the destination address
// changes, so the double that ends up in rvec[] is the same bit pattern.  What
// disappears is a blocking device->host copy followed by a blocking host->device
// copy, i.e. two implicit device-wide barriers per Runge-Kutta stage.
void rstepGPU_into(
    const thrust::device_ptr<const double>& qK,
    const thrust::device_ptr<const double>& qR,
    double t,
    const dmfe::device_vector<double>& integ,
    const dmfe::device_vector<double>& theta,
    double Gamma,
    double T0,
    StreamPool& pool,
    double* out_r) {

    const size_t len = theta.size();
    const int threads = 64;
    const int blocks = static_cast<int>((len + threads - 1) / threads);

    // NOTE: deliberately left on the LEGACY default stream.  Because the pool
    // streams are blocking streams, this launch is ordered after all pending work
    // on them and everything issued afterwards is ordered after it; it is the
    // barrier that keeps the temp0..temp4 scratch buffers, shared with
    // QKRstepGPU, free of read-after-write races.
    computeSigmaKandRKernel<<<blocks, threads, 0>>>(
        thrust::raw_pointer_cast(qK),
        thrust::raw_pointer_cast(qR),
        thrust::raw_pointer_cast(sim->device->temp0.data()),
        thrust::raw_pointer_cast(sim->device->temp1.data()),
        len
    );

    // [PERF] the two single-block ConvA reductions are one launch with a
    // (depth x 2) grid; see ConvAScalarBatchDesc in convolution/convolution.cuh.
    // Issuing them on the same stream also removes the event that used to join
    // pool[2] into pool[3].
    {
        const size_t clen = integ.size();
        const size_t cdepth = sim->device->temp1.size() / clen;
        ConvAScalarBatchDesc cb;
        cb.add(thrust::raw_pointer_cast(sim->device->temp1.data()), thrust::raw_pointer_cast(qK),
               thrust::raw_pointer_cast(sim->device->temp2.data()), t);
        cb.add(thrust::raw_pointer_cast(sim->device->temp0.data()), thrust::raw_pointer_cast(qR),
               thrust::raw_pointer_cast(sim->device->temp3.data()), t);
        ConvABatchedScalarGPU(cb, integ, theta, clen, cdepth, pool[3]);
    }

    // Fused final computation on GPU
    computeRstepResult<<<1, 1, 0, pool[3]>>>(
        thrust::raw_pointer_cast(sim->device->temp0.data()),
        thrust::raw_pointer_cast(sim->device->temp2.data()),
        thrust::raw_pointer_cast(sim->device->temp3.data()),
        thrust::raw_pointer_cast(qK),
        out_r,
        Gamma,
        T0
    );

    // [PERF] Published so that the next interpolateGPU (whose indexVecR2GPU reads
    // rvec on stream 0) can wait on it.  pool[3] is joined with pool[2] just above
    // and is itself ordered after the legacy-default-stream launch, so this single
    // event covers the whole of rstepGPU and everything that preceded it.
    cudaEventRecord(dmfe::EventPool::instance()[dmfe::EV_RSTEP_OUT], pool[3]);
}

double rstepGPU(
    const thrust::device_ptr<const double>& qK,
    const thrust::device_ptr<const double>& qR,
    double t,
    const dmfe::device_vector<double>& integ,
    const dmfe::device_vector<double>& theta,
    double Gamma,
    double T0,
    StreamPool& pool) {

    // [PERF] thin wrapper kept for the callers that really do want the value on
    // the host (appendAllGPU / replaceAllGPU, not used by the GPU integrators).
    rstepGPU_into(qK, qR, t, integ, theta, Gamma, T0, pool,
                  thrust::raw_pointer_cast(sim->device->temp4.data()));

    return sim->device->temp4[0];
}

double rstepGPU(
    const dmfe::device_vector<double>& QKv,
    const dmfe::device_vector<double>& QRv,
    const dmfe::device_vector<double>& t1grid,
    const dmfe::device_vector<double>& integ,
    const dmfe::device_vector<double>& theta,
    double Gamma,
    double T0,
    StreamPool& pool) {

    const size_t len = theta.size();
    const size_t t1len = t1grid.size();
    const double t = t1grid.back();

    const thrust::device_ptr<const double> qK = get_slice_ptr(QKv, t1len - 1, len);
    const thrust::device_ptr<const double> qR = get_slice_ptr(QRv, t1len - 1, len);

    return rstepGPU(qK, qR, t, integ, theta, Gamma, T0, pool);
}

double drstepGPU(
    const dmfe::device_vector<double>& QKv,
    const dmfe::device_vector<double>& QRv,
    const dmfe::device_vector<double>& dQKv,
    const dmfe::device_vector<double>& dQRv,
    const dmfe::device_vector<double>& t1grid,
    const dmfe::device_vector<double>& integ,
    const dmfe::device_vector<double>& theta,
    double T0) {
    
    size_t len = QKv.size();
    const double t = t1grid.back();
    dmfe::device_vector<double> sigmaK(len, 0.0);
    dmfe::device_vector<double> sigmaR(len, 0.0);
    dmfe::device_vector<double> dsigmaK(len, 0.0);
    dmfe::device_vector<double> dsigmaR(len, 0.0);

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
    dmfe::device_vector<double> convA_sigmaR_qK = ConvAGPU(sigmaR, QKv, t, integ, theta);
    dmfe::device_vector<double> convA_sigmaK_qR = ConvAGPU(sigmaK, QRv, t, integ, theta);
    dmfe::device_vector<double> convA_dsigmaR_qK = ConvAGPU(dsigmaR, QKv, t, integ, theta);
    dmfe::device_vector<double> convA_dsigmaK_qR = ConvAGPU(dsigmaK, QRv, t, integ, theta);
    dmfe::device_vector<double> convA_sigmaR_dqK = ConvAGPU(sigmaR, dQKv, t, integ, theta);
    dmfe::device_vector<double> convA_sigmaK_dqR = ConvAGPU(sigmaK, dQRv, t, integ, theta);

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
        thrust::raw_pointer_cast(sim->device->temp0.data()),
        T0
    );

    return sim->device->temp0[0];
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
        sim->device->temp0.begin(),
        sim->device->temp1.begin(),
        sim->device->temp2.begin(),
        sim->device->temp3.begin()
    ));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(
        QKv + config.len, QRv + config.len, dQKv + config.len, dQRv + config.len,
        sim->device->temp0.begin() + config.len,
        sim->device->temp1.begin() + config.len,
        sim->device->temp2.begin() + config.len,
        sim->device->temp3.begin() + config.len
    ));

    thrust::for_each(
        // [PERF] par -> par_nosync: the batched convolution below is issued on the
        // same stream, so the dependency is implicit and the trailing
        // cudaStreamSynchronize of the synchronous policy is pure host stall.
        thrust::cuda::par_nosync.on(pool[0]),
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
    // [PERF] six single-block ConvA reductions on six streams -> one launch with a
    // (depth x 6) grid on pool[0], the same stream as the thrust::for_each above,
    // so the dependency on temp0..temp3 needs no host-side barrier.  Each entry
    // keeps its own scalar t (1.0 for the first two), and the kernel body is
    // ConvAGPUKernel unchanged; see ConvAScalarBatchDesc.
    {
        const size_t clen = sim->device->integ.size();
        const size_t cdepth = sim->device->temp1.size() / clen;
        auto P = [](const dmfe::device_vector<double>& v) { return thrust::raw_pointer_cast(v.data()); };
        auto Q = [](dmfe::device_vector<double>& v) { return thrust::raw_pointer_cast(v.data()); };
        ConvAScalarBatchDesc cb;
        cb.add(P(sim->device->temp1), thrust::raw_pointer_cast(QKv),  Q(sim->device->temp4), 1.0);
        cb.add(P(sim->device->temp0), thrust::raw_pointer_cast(QRv),  Q(sim->device->temp5), 1.0);
        cb.add(P(sim->device->temp3), thrust::raw_pointer_cast(QKv),  Q(sim->device->temp6), t);
        cb.add(P(sim->device->temp2), thrust::raw_pointer_cast(QRv),  Q(sim->device->temp7), t);
        cb.add(P(sim->device->temp1), thrust::raw_pointer_cast(dQKv), Q(sim->device->temp8), t);
        cb.add(P(sim->device->temp0), thrust::raw_pointer_cast(dQRv), Q(sim->device->temp9), t);
        ConvABatchedScalarGPU(cb, sim->device->integ, sim->device->theta, clen, cdepth, pool[0]);
    }

    // [PERF] cudaDeviceSynchronize() removed: computeDrstep2Result below is launched
    // on the LEGACY default stream, which is by definition ordered after all pending
    // work on every blocking stream, and pool[1..6] are blocking streams (created
    // with plain cudaStreamCreate).  The barrier was therefore redundant for
    // correctness and only stalled the host once per accepted step.

    // Compute final result
    computeDrstep2Result<<<1, 1, 0>>>(
        thrust::raw_pointer_cast(sim->device->temp0.data()),
        thrust::raw_pointer_cast(sim->device->temp2.data()),
        thrust::raw_pointer_cast(sim->device->temp4.data()),
        thrust::raw_pointer_cast(sim->device->temp5.data()),
        thrust::raw_pointer_cast(sim->device->temp6.data()),
        thrust::raw_pointer_cast(sim->device->temp7.data()),
        thrust::raw_pointer_cast(sim->device->temp8.data()),
        thrust::raw_pointer_cast(sim->device->temp9.data()),
        thrust::raw_pointer_cast(QKv),
        thrust::raw_pointer_cast(dQKv),
        thrust::raw_pointer_cast(sim->device->temp1.data()),
        T0
    );

    return sim->device->temp1[0];
}

double energyGPU(
    const DeviceSimulationData& dev,
    double T0) {

    const auto& QKv = dev.QKv;
    const auto& QRv = dev.QRv;
    const auto& t1grid = dev.t1grid;
    const auto& integ = dev.integ;
    const auto& theta = dev.theta;
    
    size_t len = theta.size();
    size_t t1len = t1grid.size();
    // [PERF] host mirror instead of a blocking one-element D2H copy of
    // t1grid.back(); see core/t1_cache.hpp.
    const double t = dmfe::t1LastGPU();

    thrust::device_ptr<double> qK = get_slice_ptr(QKv, t1len - 1, len);
    thrust::device_ptr<double> qR = get_slice_ptr(QRv, t1len- 1, len);

    int threads = 64;
    int blocks = (len + threads - 1) / threads;

    computeSigmaKandRKernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(qK),
        thrust::raw_pointer_cast(qR),
        thrust::raw_pointer_cast(dev.temp0.data()),
        thrust::raw_pointer_cast(dev.temp1.data()),
        len
    );

    // Compute convolution results
    ConvAGPU_Stream(dev.temp0, qR, dev.temp2, t, integ, theta);

    // Compute final result
    double result = - (dev.temp2[0] + flambda(qK[0]) / T0);

    return move(result);
}

// GPU append/replace functions (CPU versions are in time_steps.cpp)
void appendGPU(dmfe::device_vector<double>& dest,
                                        const dmfe::device_vector<double>& src, double scale) {
    size_t required_size = dest.size() + src.size();

    if (dest.capacity() < required_size) {
        // Allocate a new vector with more capacity (e.g., double the required size)
        size_t new_capacity = std::max(required_size, 2 * dest.capacity());
        dmfe::device_vector<double> tmp;
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

void appendGPU_ptr(dmfe::device_vector<double>& dest,
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
    const dmfe::device_vector<double>& qK,
    const dmfe::device_vector<double>& qR,
    const dmfe::device_vector<double>& dqK,
    const dmfe::device_vector<double>& dqR,
    const double dr,
    const double t,
    StreamPool& pool)
{
    size_t length = qK.size();
    if (length != qR.size() || length != dqK.size() || length != dqR.size()) {
        throw invalid_argument("All input vectors must have the same size.");
    }

    // 1) update sim->device->t1grid and sim->device->delta_t_ratio
    sim->device->t1grid.push_back(t);
    dmfe::t1LastSet(t);   // [PERF] see core/t1_cache.hpp
    size_t idx = sim->device->t1grid.size() - 1;
    double tdiff = sim->device->t1grid[idx] - sim->device->t1grid[idx - 1];
    if (idx > 1) {
        double prev = sim->device->t1grid[idx - 1] - sim->device->t1grid[idx - 2];
        sim->device->delta_t_ratio.push_back(tdiff / prev);
    }
    else {
        sim->device->delta_t_ratio.push_back(0.0);
    }

    appendGPU(sim->device->QKv,qK);
    appendGPU(sim->device->QRv,qR);
    appendGPU(sim->device->dQKv, dqK, tdiff);
    appendGPU(sim->device->dQRv, dqR, tdiff);

    // 2) finally update drvec and rvec
    sim->device->drvec.push_back(tdiff * dr);

    const thrust::device_ptr<const double> lastQK = thrust::device_pointer_cast(qK.data());
    const thrust::device_ptr<const double> lastQR = thrust::device_pointer_cast(qR.data());
    sim->device->rvec.push_back(rstepGPU(lastQK, lastQR, t, sim->device->integ, sim->device->theta, config.Gamma, config.T0, pool));
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
    const size_t idx = sim->device->t1grid.size();

    // Keep existing resize cadence: grow by exactly one time-step when needed
    if (sim->device->t1grid.capacity() < idx + 1) {
        sim->device->t1grid.reserve(idx + 1000);
        sim->device->delta_t_ratio.reserve(idx + 1000);
    }

    // Ensure scalar timelines have space for the new entry
    sim->device->t1grid.resize(idx + 1);
    sim->device->delta_t_ratio.resize(idx + 1);
    sim->device->drvec.resize(idx + 1);
    sim->device->rvec.resize(idx + 1);

    // History buffers: compute offset and resize by +len (no change in frequency)
    const size_t offset = sim->device->QKv.size();
    if (sim->device->QKv.capacity() < offset + len) sim->device->QKv.reserve(offset + 1000 * len);
    if (sim->device->QRv.capacity() < offset + len) sim->device->QRv.reserve(offset + 1000 * len);
    if (sim->device->dQKv.capacity() < offset + len) sim->device->dQKv.reserve(offset + 1000 * len);
    if (sim->device->dQRv.capacity() < offset + len) sim->device->dQRv.reserve(offset + 1000 * len);
    sim->device->QKv.resize(offset + len);
    sim->device->QRv.resize(offset + len);
    sim->device->dQKv.resize(offset + len);
    sim->device->dQRv.resize(offset + len);

    // Launch fused history append (1 kernel instead of 4)
    const int threads = 64;
    const int blocks  = static_cast<int>((len + threads - 1) / threads);
    append4Kernel<<<blocks, threads, 0, pool[0]>>>(
        thrust::raw_pointer_cast(qK),
        thrust::raw_pointer_cast(qR),
        thrust::raw_pointer_cast(dqK),
        thrust::raw_pointer_cast(dqR),
        thrust::raw_pointer_cast(sim->device->QKv.data()),
        thrust::raw_pointer_cast(sim->device->QRv.data()),
        thrust::raw_pointer_cast(sim->device->dQKv.data()),
        thrust::raw_pointer_cast(sim->device->dQRv.data()),
        thrust::raw_pointer_cast(sim->device->t1grid.data()),
        thrust::raw_pointer_cast(sim->device->delta_t_ratio.data()),
        thrust::raw_pointer_cast(sim->device->drvec.data()),
        idx,
        offset,
        len,
        t,
        dr);

    // [PERF] append4Kernel has just stored this very double at t1grid[idx]; keep
    // the host mirror in step so that t1grid.back() never has to be read back.
    dmfe::t1LastSet(t);

    // [PERF] the time scalars (t1grid[idx], delta_t_ratio[idx], drvec[idx]) are now
    // written by the append/replace kernel above; the separate <<<1,1>>> launch on
    // pool[1] is gone.

    // Compute r(t) and store it in d_rvec[idx]
    // [PERF] was `sim->device->rvec[idx] = rstepGPU(...)`, i.e. one double copied
    // device->host (blocking) and immediately copied back host->device (blocking).
    // rvec has just been resized, so the address is valid for the whole launch.
    rstepGPU_into(qK, qR, t, sim->device->integ, sim->device->theta,
                  config.Gamma, config.T0, pool,
                  thrust::raw_pointer_cast(sim->device->rvec.data()) + idx);
}

void replaceAllGPU(
    const dmfe::device_vector<double>& qK,
    const dmfe::device_vector<double>& qR,
    const dmfe::device_vector<double>& dqK,
    const dmfe::device_vector<double>& dqR,
    const double dr,
    const double t,
    StreamPool& pool)
{   
    // Replace the existing values in the vectors with the new values
    size_t replaceLength = qK.size();
    size_t length = sim->device->QKv.size() - replaceLength;
    if (replaceLength != qR.size() || replaceLength != dqK.size() || replaceLength != dqR.size()) {
        throw invalid_argument("All input vectors must have the same size.");
    }
    {
        sim->device->t1grid.back() = t;
        dmfe::t1LastSet(t);   // [PERF] see core/t1_cache.hpp
        double tdiff = (sim->device->t1grid[sim->device->t1grid.size() - 1] - sim->device->t1grid[sim->device->t1grid.size() - 2]);

        if (sim->device->t1grid.size() > 2) {
            sim->device->delta_t_ratio.back() = tdiff /
                (sim->device->t1grid[sim->device->t1grid.size() - 2] - sim->device->t1grid[sim->device->t1grid.size() - 3]);
        }
        else {
            sim->device->delta_t_ratio.back() = 0.0;
        }

        thrust::copy(qK.begin(), qK.end(), sim->device->QKv.begin() + length);
        thrust::copy(qR.begin(), qR.end(), sim->device->QRv.begin() + length);
        thrust::transform(dqK.begin(), dqK.end(), sim->device->dQKv.begin() + length, [tdiff] __device__ (double x) { return tdiff * x; });
        thrust::transform(dqR.begin(), dqR.end(), sim->device->dQRv.begin() + length, [tdiff] __device__ (double x) { return tdiff * x; });

        sim->device->drvec.back() = tdiff * dr;

        const size_t idx = sim->device->t1grid.size() - 1;
        const thrust::device_ptr<const double> lastQK = thrust::device_pointer_cast(qK.data());
        const thrust::device_ptr<const double> lastQR = thrust::device_pointer_cast(qR.data());
        sim->device->rvec.back() = rstepGPU(lastQK, lastQR, t, sim->device->integ, sim->device->theta, config.Gamma, config.T0, pool);
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
    const size_t gridSize = sim->device->t1grid.size();
    const size_t idx = gridSize - 1;
    const size_t offset = sim->device->QKv.size() - len;

    const int threads = 64;
    const int blocks  = static_cast<int>((len + threads - 1) / threads);

    // Fused replace of 4 arrays with dt scaling computed on device
    replace4Kernel<<<blocks, threads, 0, pool[0]>>>(
        thrust::raw_pointer_cast(qK),
        thrust::raw_pointer_cast(qR),
        thrust::raw_pointer_cast(dqK),
        thrust::raw_pointer_cast(dqR),
        thrust::raw_pointer_cast(sim->device->QKv.data()),
        thrust::raw_pointer_cast(sim->device->QRv.data()),
        thrust::raw_pointer_cast(sim->device->dQKv.data()),
        thrust::raw_pointer_cast(sim->device->dQRv.data()),
        thrust::raw_pointer_cast(sim->device->t1grid.data()),
        thrust::raw_pointer_cast(sim->device->delta_t_ratio.data()),
        thrust::raw_pointer_cast(sim->device->drvec.data()),
        idx,
        offset,
        len,
        t,
        dr);

    // [PERF] replace4Kernel has just stored this very double at t1grid[idx].
    dmfe::t1LastSet(t);

    // [PERF] the time scalars (t1grid[idx], delta_t_ratio[idx], drvec[idx]) are now
    // written by the append/replace kernel above; the separate <<<1,1>>> launch on
    // pool[1] is gone.

    // Update r(t) at the last position
    // [PERF] see appendAllGPU_ptr: the round trip through the host is gone.
    rstepGPU_into(qK, qR, t, sim->device->integ, sim->device->theta,
                  config.Gamma, config.T0, pool,
                  thrust::raw_pointer_cast(sim->device->rvec.data())
                      + (sim->device->rvec.size() - 1));
}
