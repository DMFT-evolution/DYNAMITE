#ifndef TIME_STEPS_HPP
#define TIME_STEPS_HPP

#include "core/config_build.hpp"
#include <vector>
#include "core/stream_pool.hpp"

#if DMFE_WITH_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#endif

// Utility functions for extracting last entries
std::vector<double> getLastLenEntries(const std::vector<double>& vec, size_t len);

#if DMFE_WITH_CUDA
thrust::device_vector<double> getLastLenEntriesGPU(const thrust::device_vector<double>& vec, size_t len);

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

// CPU time-step functions
std::vector<double> QKstep();
std::vector<double> QRstep();
double rstep();
double drstep();
double drstep2(const std::vector<double>& qK, const std::vector<double>& qR, 
               const std::vector<double>& dqK, const std::vector<double>& dqR, const double t);

#if DMFE_WITH_CUDA
// GPU time-step functions
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
    StreamPool& pool);

thrust::device_vector<double> QRstepGPU(
    const thrust::device_vector<double>& QRv,
    const thrust::device_vector<double>& rInt,
    const thrust::device_vector<double>& SigmaRA2int,
    const thrust::device_vector<double>& QRB2int,
    const thrust::device_vector<double>& QRA2int,
    const thrust::device_vector<double>& SigmaRB2int,
    const thrust::device_vector<double>& t1grid,
    const thrust::device_vector<double>& theta,
    StreamPool& pool);

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
    StreamPool& pool);

double rstepGPU(
    const thrust::device_vector<double>& QKv,
    const thrust::device_vector<double>& QRv,
    const thrust::device_vector<double>& t1grid,
    const thrust::device_vector<double>& integ,
    const thrust::device_vector<double>& theta,
    double Gamma,
    double T0,
    StreamPool& pool);

double drstepGPU(
    const thrust::device_vector<double>& QKv,
    const thrust::device_vector<double>& QRv,
    const thrust::device_vector<double>& dQKv,
    const thrust::device_vector<double>& dQRv,
    const thrust::device_vector<double>& t1grid,
    const thrust::device_vector<double>& integ,
    const thrust::device_vector<double>& theta,
    double T0);

double drstep2GPU(
    const thrust::device_ptr<double>& QKv,
    const thrust::device_ptr<double>& QRv,
    const thrust::device_ptr<double>& dQKv,
    const thrust::device_ptr<double>& dQRv,
    const double t,
    const double T0,
    StreamPool& pool);

double energyGPU(
    const thrust::device_vector<double>& QKv,
    const thrust::device_vector<double>& QRv,
    const thrust::device_vector<double>& t1grid,
    const thrust::device_vector<double>& integ,
    const thrust::device_vector<double>& theta,
    double T0);

// Helper functions
void QRstepFused(const thrust::device_ptr<double>& qR,
                 const thrust::device_vector<double>& theta,
                 const thrust::device_vector<double>& conv1,
                 const thrust::device_vector<double>& conv2,
                 const thrust::device_vector<double>& r,
                 double* out,
                 cudaStream_t stream = 0);

void appendGPU(thrust::device_vector<double>& dest,
               const thrust::device_vector<double>& src, 
               double scale = 1.0);

void appendGPU_ptr(thrust::device_vector<double>& dest,
                   const thrust::device_ptr<double>& src, 
                   double size, 
                   double scale = 1.0, 
                   cudaStream_t stream = 0);

void appendAllGPU(
    const thrust::device_vector<double>& qK,
    const thrust::device_vector<double>& qR,
    const thrust::device_vector<double>& dqK,
    const thrust::device_vector<double>& dqR,
    const double dr,
    const double t,
    StreamPool& pool);

void appendAllGPU_ptr(
    const thrust::device_ptr<double>& qK,
    const thrust::device_ptr<double>& qR,
    const thrust::device_ptr<double>& dqK,
    const thrust::device_ptr<double>& dqR,
    const double dr,
    const double t,
    const size_t len,
    StreamPool& pool);

void replaceAllGPU(
    const thrust::device_vector<double>& qK,
    const thrust::device_vector<double>& qR,
    const thrust::device_vector<double>& dqK,
    const thrust::device_vector<double>& dqR,
    const double dr,
    const double t,
    StreamPool& pool);

void replaceAllGPU_ptr(
    const thrust::device_ptr<double>& qK,
    const thrust::device_ptr<double>& qR,
    const thrust::device_ptr<double>& dqK,
    const thrust::device_ptr<double>& dqR,
    const double dr,
    const double t,
    const size_t len,
    StreamPool& pool);
#endif // DMFE_WITH_CUDA

// CPU Append/Replace functions (available in both CUDA and CPU-only modes)
void appendAll(const std::vector<double>& qK,
               const std::vector<double>& qR,
               const std::vector<double>& dqK,
               const std::vector<double>& dqR,
               const double dr,
               const double t);

void replaceAll(const std::vector<double>& qK, 
                const std::vector<double>& qR, 
                const std::vector<double>& dqK, 
                const std::vector<double>& dqR, 
                const double dr, 
                const double t);

#endif // TIME_STEPS_HPP
