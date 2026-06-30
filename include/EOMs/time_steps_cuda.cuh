#ifndef TIME_STEPS_CUDA_CUH
#define TIME_STEPS_CUDA_CUH

#include "time_steps_cpu.hpp"

#include "core/config_build.hpp"

#if DMFE_WITH_CUDA

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include "simulation/device_simulation_data.hpp"

#include "core/device_vector.hpp"
#include "core/stream_pool.hpp"

// ============================================================================
// GPU interface
// ============================================================================

dmfe::device_vector<double>
getLastLenEntriesGPU(const dmfe::device_vector<double>& vec,
                     size_t len);

dmfe::device_vector<double>
QKstepGPU(
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
    StreamPool& pool);

dmfe::device_vector<double>
QRstepGPU(
    const dmfe::device_vector<double>& QRv,
    const dmfe::device_vector<double>& rInt,
    const dmfe::device_vector<double>& SigmaRA2int,
    const dmfe::device_vector<double>& QRB2int,
    const dmfe::device_vector<double>& QRA2int,
    const dmfe::device_vector<double>& SigmaRB2int,
    const dmfe::device_vector<double>& t1grid,
    const dmfe::device_vector<double>& theta,
    StreamPool& pool);

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
    StreamPool& pool);

double rstepGPU(
    const dmfe::device_vector<double>& QKv,
    const dmfe::device_vector<double>& QRv,
    const dmfe::device_vector<double>& t1grid,
    const dmfe::device_vector<double>& integ,
    const dmfe::device_vector<double>& theta,
    double Gamma,
    double T0,
    StreamPool& pool);

double drstepGPU(
    const dmfe::device_vector<double>& QKv,
    const dmfe::device_vector<double>& QRv,
    const dmfe::device_vector<double>& dQKv,
    const dmfe::device_vector<double>& dQRv,
    const dmfe::device_vector<double>& t1grid,
    const dmfe::device_vector<double>& integ,
    const dmfe::device_vector<double>& theta,
    double T0);

double drstep2GPU(
    const thrust::device_ptr<double>& QKv,
    const thrust::device_ptr<double>& QRv,
    const thrust::device_ptr<double>& dQKv,
    const thrust::device_ptr<double>& dQRv,
    double t,
    double T0,
    StreamPool& pool);

double energyGPU(
    const DeviceSimulationData& sim,
    double T0);

void QRstepFused(
    const thrust::device_ptr<double>& qR,
    const dmfe::device_vector<double>& theta,
    const dmfe::device_vector<double>& conv1,
    const dmfe::device_vector<double>& conv2,
    const dmfe::device_vector<double>& r,
    double* out,
    cudaStream_t stream = 0);

void appendGPU(dmfe::device_vector<double>& dest,
               const dmfe::device_vector<double>& src,
               double scale = 1.0);

void appendGPU_ptr(dmfe::device_vector<double>& dest,
                   const thrust::device_ptr<double>& src,
                   double size,
                   double scale = 1.0,
                   cudaStream_t stream = 0);

void appendAllGPU(
    const dmfe::device_vector<double>& qK,
    const dmfe::device_vector<double>& qR,
    const dmfe::device_vector<double>& dqK,
    const dmfe::device_vector<double>& dqR,
    double dr,
    double t,
    StreamPool& pool);

void appendAllGPU_ptr(
    const thrust::device_ptr<double>& qK,
    const thrust::device_ptr<double>& qR,
    const thrust::device_ptr<double>& dqK,
    const thrust::device_ptr<double>& dqR,
    double dr,
    double t,
    size_t len,
    StreamPool& pool);

void replaceAllGPU(
    const dmfe::device_vector<double>& qK,
    const dmfe::device_vector<double>& qR,
    const dmfe::device_vector<double>& dqK,
    const dmfe::device_vector<double>& dqR,
    double dr,
    double t,
    StreamPool& pool);

void replaceAllGPU_ptr(
    const thrust::device_ptr<double>& qK,
    const thrust::device_ptr<double>& qR,
    const thrust::device_ptr<double>& dqK,
    const thrust::device_ptr<double>& dqR,
    double dr,
    double t,
    size_t len,
    StreamPool& pool);


// Forward declaration of kernel functions
__global__ void FusedQRKernel(
    const double* __restrict__ qR,
    const double* __restrict__ theta,
    const double* __restrict__ conv1,
    const double* __restrict__ conv2,
    const double* __restrict__ r,
    double* __restrict__ out,
    size_t len);

__global__ void computeRstepResult(
    const double* __restrict__ temp0,
    const double* __restrict__ temp2,
    const double* __restrict__ temp3,
    const double* __restrict__ qK,
    double* __restrict__ result,
    double Gamma,
    double T0);

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
    double T0);

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
    double T0);

__global__ void computeCopy(
    const double* __restrict__ src,
    double* __restrict__ dest,
    size_t offset,
    size_t len,
    double factor = 1.0);

#endif // DMFE_WITH_CUDA
#endif // TIME_STEPS_CUDA_CUH