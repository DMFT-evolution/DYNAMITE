#ifndef RUNGE_KUTTA_HPP
#define RUNGE_KUTTA_HPP

#include "core/config_build.hpp"
#include <vector>
#include "core/stream_pool.hpp"

#if DMFE_WITH_CUDA
// Forward declarations for kernel functions used by Runge-Kutta methods
__global__ void computeError(const double* __restrict__ gKfinal,
                            const double* __restrict__ gKe,
                            const double* __restrict__ gRfinal,
                            const double* __restrict__ gRe,
                            double* __restrict__ result,
                            size_t len);
#endif

// CPU Runge-Kutta methods
double SSPRK104();
double RK54();

// Runge-Kutta initialization functions (work for both CPU and GPU)
void init_RK54GPU();
void init_SSPRK104GPU();
void init_SERK2(int q);

double update();

#if DMFE_WITH_CUDA
// GPU Runge-Kutta methods
double RK54GPU(StreamPool* pool = nullptr);
double SSPRK104GPU(StreamPool* pool = nullptr);
double SERK2GPU(int q, StreamPool* pool = nullptr);

// Helper functions for method selection
double updateGPU(StreamPool* pool = nullptr);
#endif

// SERK coefficient generation functions
long double chebyshevT_ld(int n, long double x);
long double chebyshevU_ld(int n, long double x);
std::vector<long double> gaussianElimination_ld(std::vector<std::vector<long double>> A, std::vector<long double> b);
std::vector<double> SERKcoeffs(int q);

#endif // RUNGE_KUTTA_HPP
