#include "time_steps.hpp"
#include "globals.hpp"
#include "config.hpp"
#include "math_ops.hpp"
#include "vector_utils.hpp"
#include "convolution.hpp"
#include "compute_utils.hpp"
#include "math_sigma.hpp"
#include "io_utils.hpp"
#include <vector>
#include <omp.h>

using namespace std;

// External declaration for global config variable
extern SimulationConfig config;

// Utility functions for extracting last entries
vector<double> getLastLenEntries(const vector<double>& vec, size_t len) {
    if (len > vec.size()) {
        throw invalid_argument("len is greater than the size of the vector.");
    }
    return vector<double>(vec.end() - len, vec.end());
}

// CPU time-step functions
vector<double> QKstep()
{
    vector<double> temp(config.len, 0.0);
    vector<double> qK = getLastLenEntries(sim->h_QKv, config.len);
    vector<double> qR = getLastLenEntries(sim->h_QRv, config.len);
    #pragma omp parallel for
    for (size_t i = 0; i < sim->h_QKB1int.size(); i += config.len) {
        temp[i / config.len] = sim->h_QKB1int[i];
    }
    vector<double> d1qK = (temp* (Dflambda(sim->h_QKv[sim->h_QKv.size() - config.len]) / config.T0)) + (qK * (-sim->h_rInt.back())) +
    ConvR(sim->h_SigmaRA2int, sim->h_QKB2int, sim->h_t1grid.back()) + ConvA(sim->h_SigmaRA1int, sim->h_QKB1int, sim->h_t1grid.back()) +
    ConvA(sim->h_SigmaKA1int, sim->h_QRB1int, sim->h_t1grid.back());
    #pragma omp parallel for
    for (size_t i = 0; i < sim->h_QKB1int.size(); i += config.len) {
        temp[i / config.len] = Dflambda(sim->h_QKB1int[i]);
    }
    vector<double> d2qK = (temp * (sim->h_QKv[sim->h_QKv.size() - config.len] / config.T0)) + (qR * (2 * config.Gamma)) +
    ConvR(sim->h_QRA2int, sim->h_SigmaKB2int, sim->h_t1grid.back()) + ConvA(sim->h_QRA1int, sim->h_SigmaKB1int, sim->h_t1grid.back()) +
    ConvA(sim->h_QKA1int, sim->h_SigmaRB1int, sim->h_t1grid.back()) - (qK * sim->h_rInt);
    return d1qK + (d2qK * sim->h_theta);
}

void replaceAll(const vector<double>& qK, const vector<double>& qR, const vector<double>& dqK, const vector<double>& dqR, const double dr, const double t)
{
    // Replace the existing values in the vectors with the new values
    size_t replaceLength = qK.size();
    size_t length = sim->h_QKv.size() - replaceLength;
    if (replaceLength != qR.size() || replaceLength != dqK.size() || replaceLength != dqR.size()) {
        throw invalid_argument("All input vectors must have the same size.");
    }
    {
        sim->h_t1grid.back() = t;
        double tdiff = (sim->h_t1grid[sim->h_t1grid.size() - 1] - sim->h_t1grid[sim->h_t1grid.size() - 2]);

        if (sim->h_t1grid.size() > 2) {
            sim->h_delta_t_ratio.back() = tdiff /
                (sim->h_t1grid[sim->h_t1grid.size() - 2] - sim->h_t1grid[sim->h_t1grid.size() - 3]);
        }
        else {
            sim->h_delta_t_ratio.back() = 0.0;
        }

        #pragma omp parallel for
        for (size_t i = 0; i < replaceLength; i++)
        {
            sim->h_QKv[length + i] = qK[i];
            sim->h_QRv[length + i] = qR[i];
            sim->h_dQKv[length + i] = tdiff * dqK[i];
            sim->h_dQRv[length + i] = tdiff * dqR[i];
        }

        sim->h_drvec.back() = tdiff * dr;
        sim->h_rvec.back() = rstep();
    }
}
