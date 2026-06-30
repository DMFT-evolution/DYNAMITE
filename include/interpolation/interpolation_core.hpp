#ifndef INTERPOLATION_CORE_HPP
#define INTERPOLATION_CORE_HPP

#include "core/config_build.hpp"
#include <vector>
#include "core/stream_pool.hpp"

#if DMFE_WITH_CUDA
#include <cuda_runtime.h>
#endif

// CPU interpolation function
void interpolate(const std::vector<double>& posB1xIn = {}, 
                 const std::vector<double>& posB2xIn = {},
                 const bool same = false);

#if DMFE_WITH_CUDA
// GPU interpolation function
void interpolateGPU(const double* posB1xIn = nullptr,
                    const double* posB2xIn = nullptr,
                    const bool same = false,
                    StreamPool* pool = nullptr);

// Helper functions
void diffNfloor(const double* posB1x,
                size_t* Floor,
                double* diff,
                size_t len,
                cudaStream_t stream = 0);

// CUDA kernel
__global__ void diffNfloorKernel(const double* __restrict__ posB1x,
                                 size_t* __restrict__ Floor,
                                 double* __restrict__ diff,
                                 size_t len);
#endif // DMFE_WITH_CUDA

#endif // INTERPOLATION_CORE_HPP
