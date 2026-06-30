#pragma once
#include "math/math_sigma.hpp"

#if DMFE_WITH_CUDA
#include "core/device_vector.hpp"

// GPU kernel and host wrappers for SigmaK/SigmaR evaluations
__global__ void computeSigmaKandRKernel(const double* __restrict__ qK,
										const double* __restrict__ qR,
										double* __restrict__ sigmaK,
										double* __restrict__ sigmaR,
										size_t len);

// Sigma GPU function declarations
void SigmaKGPU(const dmfe::device_vector<double>& qk, dmfe::device_vector<double>& result, cudaStream_t stream = 0);
void SigmaRGPU(const dmfe::device_vector<double>& qk, const dmfe::device_vector<double>& qr, dmfe::device_vector<double>& result, cudaStream_t stream = 0);
#endif // DMFE_WITH_CUDA
