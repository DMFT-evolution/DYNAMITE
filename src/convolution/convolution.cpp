#include "convolution/convolution.hpp"
#include "core/globals.hpp"

#include <omp.h>

std::vector<double> ConvA(const std::vector<double>& f, const std::vector<double>& g, const double t)
{
    size_t length = sim->host->integ.size();
    size_t depth = f.size() / length;
    std::vector<double> out(depth, 0.0);
    if (depth == 1)
    {
        double temp = 0.0;
        #pragma omp parallel for reduction(+:temp)
        for (size_t j = 0; j < length; j++)
        {
            temp += t * sim->host->integ[j] * f[j] * g[j];
        }
        out[0] = temp;
    }
    else
    {
        #pragma omp parallel for
        for (size_t j = 0; j < depth; j++)
        {
            for (size_t i = 0; i < length; i++)
            {
                out[j] += sim->host->integ[i] * f[j * length + i] * g[j * length + i];
            }
            out[j] *= t * sim->host->theta[j];
        }
    }
    return out;
}

std::vector<double> ConvR(const std::vector<double>& f, const std::vector<double>& g, const double t)
{
    size_t length = sim->host->integ.size();
    size_t depth = f.size() / length;
    std::vector<double> out(length, 0.0);
    #pragma omp parallel for
    for (size_t j = 0; j < length; j++)
    {
        for (size_t i = 0; i < depth; i++)
        {
            out[j] += sim->host->integ[i] * f[j * length + i] * g[j * length + i];
        }
        out[j] *= t * (1 - sim->host->theta[j]);
    }
    return out;
}
