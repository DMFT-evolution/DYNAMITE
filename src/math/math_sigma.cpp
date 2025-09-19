#include "math_sigma.hpp"
#include "math_ops.hpp"
#include <vector>
#include <omp.h>

// CPU versions with OpenMP parallelization
void SigmaR(const std::vector<double>& qk, const std::vector<double>& qr, std::vector<double>& result)
{
    #pragma omp parallel for
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDflambda(qk[i]) * qr[i];
    }
}

std::vector<double> SigmaK10(const std::vector<double>& qk)
{
    std::vector<double> result(qk.size());
    #pragma omp parallel for
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDflambda(qk[i]);
    }
    return std::move(result);
}

std::vector<double> SigmaR10(const std::vector<double>& qk, const std::vector<double>& qr)
{
    std::vector<double> result(qk.size());
    #pragma omp parallel for
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDDflambda(qk[i]) * qr[i];
    }
    return std::move(result);
}

std::vector<double> SigmaK01(const std::vector<double>& qk)
{
    std::vector<double> result(qk.size());
    #pragma omp parallel for
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = 0.0;
    }
    return std::move(result);
}

std::vector<double> SigmaR01(const std::vector<double>& qk, const std::vector<double>& qr)
{
    std::vector<double> result(qk.size());
    #pragma omp parallel for
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = DDflambda(qk[i]);
    }
    return std::move(result);
}

// CPU version of SigmaK with OpenMP parallelization
void SigmaK(const std::vector<double>& qk, std::vector<double>& result)
{
    #pragma omp parallel for if(qk.size() > 1000)
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = Dflambda(qk[i]);
    }
}
