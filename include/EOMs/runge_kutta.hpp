#ifndef RUNGE_KUTTA_HPP
#define RUNGE_KUTTA_HPP

#include "core/config_build.hpp"
#include <vector>


// CPU Runge-Kutta methods
double SSPRK104();
double RK54();

// Runge-Kutta initialization functions (work for both CPU and GPU)
void init_RK54GPU();
void init_SSPRK104GPU();
void init_SERK2(int q);

double updateCPU();

// SERK coefficient generation functions
long double chebyshevT_ld(int n, long double x);
long double chebyshevU_ld(int n, long double x);
std::vector<long double> gaussianElimination_ld(std::vector<std::vector<long double>> A, std::vector<long double> b);
std::vector<double> SERKcoeffs(int q);

#endif //RUNGE_KUTTA_HPP
