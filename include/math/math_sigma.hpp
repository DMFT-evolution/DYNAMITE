#pragma once
#include "core/config_build.hpp"
#include <cstddef>
#include <vector>

#if DMFE_WITH_CUDA
#include <thrust/device_vector.h>

// GPU kernel and host wrappers for SigmaK/SigmaR evaluations
__global__ void computeSigmaKandRKernel(const double* __restrict__ qK,
										const double* __restrict__ qR,
										double* __restrict__ sigmaK,
										double* __restrict__ sigmaR,
										size_t len);

// Sigma GPU function declarations
void SigmaKGPU(const thrust::device_vector<double>& qk, thrust::device_vector<double>& result, cudaStream_t stream = 0);
void SigmaRGPU(const thrust::device_vector<double>& qk, const thrust::device_vector<double>& qr, thrust::device_vector<double>& result, cudaStream_t stream = 0);
#endif // DMFE_WITH_CUDA

// Sigma CPU function declarations
void SigmaK(const std::vector<double>& qk, std::vector<double>& result);
void SigmaR(const std::vector<double>& qk, const std::vector<double>& qr, std::vector<double>& result);
std::vector<double> SigmaK10(const std::vector<double>& qk);
std::vector<double> SigmaR10(const std::vector<double>& qk, const std::vector<double>& qr);
std::vector<double> SigmaK01(const std::vector<double>& qk);
std::vector<double> SigmaR01(const std::vector<double>& qk, const std::vector<double>& qr);
