#ifndef RUNGE_KUTTA_HPP
#define RUNGE_KUTTA_HPP

#include <vector>
#include "stream_pool.hpp"

// Forward declarations for kernel functions used by Runge-Kutta methods
__global__ void computeError(const double* __restrict__ gKfinal,
                            const double* __restrict__ gKe,
                            const double* __restrict__ gRfinal,
                            const double* __restrict__ gRe,
                            double* __restrict__ result,
                            size_t len);

// CPU Runge-Kutta methods
double SSPRK104();
double RK54();

// GPU Runge-Kutta initialization functions
void init_RK54GPU();
void init_SSPRK104GPU();
void init_SERK2(int q);

// GPU Runge-Kutta methods
double RK54GPU(StreamPool* pool = nullptr);
double SSPRK104GPU(StreamPool* pool = nullptr);
double SERK2GPU(int q, StreamPool* pool = nullptr);

// Helper functions for method selection
double update();
double updateGPU(StreamPool* pool = nullptr);

// SERK coefficient generation functions
long double chebyshevT_ld(int n, long double x);
long double chebyshevU_ld(int n, long double x);
std::vector<long double> gaussianElimination_ld(std::vector<std::vector<long double>> A, std::vector<long double> b);
std::vector<double> SERKcoeffs(int q);

#endif // RUNGE_KUTTA_HPP
