#ifndef RUNGE_KUTTA_CUH
#define RUNGE_KUTTA_CUH

#include "core/config_build.hpp"
#include "core/stream_pool.hpp"
#include "EOMs/runge_kutta.hpp"

#if DMFE_WITH_CUDA
// Forward declarations for kernel functions used by Runge-Kutta methods
__global__ void computeError(const double* __restrict__ gKfinal,
                            const double* __restrict__ gKe,
                            const double* __restrict__ gRfinal,
                            const double* __restrict__ gRe,
                            double* __restrict__ result,
                            size_t len);

// GPU Runge-Kutta methods
double RK54GPU(StreamPool* pool = nullptr);
double SSPRK104GPU(StreamPool* pool = nullptr);
double SERK2GPU(int q, StreamPool* pool = nullptr);

// Helper functions for method selection
double updateGPU(StreamPool* pool = nullptr);
#endif // DMFE_WITH_CUDA

#endif // RUNGE_KUTTA_CUH