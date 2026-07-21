#include "EOMs/time_steps.hpp"
#include "core/globals.hpp"

#include "core/config.hpp"
#include "math/math_ops.hpp"
#include "core/vector_utils.hpp"
#include "convolution/convolution.hpp"
#include "core/compute_utils.hpp"
#include "math/math_sigma.hpp"
#include "io/io_utils.hpp"
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
vector<double> QKstep(const vector<double>& qK, const vector<double>& qR)
{
    vector<double> temp(config.len, 0.0);
    #pragma omp parallel for
    for (size_t i = 0; i < sim->host->QKB1int.size(); i += config.len) {
        temp[i / config.len] = sim->host->QKB1int[i];
    }
    vector<double> d1qK = (temp* (Dflambda(qK[0]) / config.T0)) + (qK * (-sim->host->rInt.back())) +
    ConvR(sim->host->SigmaRA2int, sim->host->QKB2int, sim->host->t1grid.back()) + ConvA(sim->host->SigmaRA1int, sim->host->QKB1int, sim->host->t1grid.back()) +
    ConvA(sim->host->SigmaKA1int, sim->host->QRB1int, sim->host->t1grid.back());
    #pragma omp parallel for
    for (size_t i = 0; i < sim->host->QKB1int.size(); i += config.len) {
        temp[i / config.len] = Dflambda(sim->host->QKB1int[i]);
    }
    vector<double> d2qK = (temp * (qK[0] / config.T0)) + (qR * (2 * config.Gamma)) +
    ConvR(sim->host->QRA2int, sim->host->SigmaKB2int, sim->host->t1grid.back()) + ConvA(sim->host->QRA1int, sim->host->SigmaKB1int, sim->host->t1grid.back()) +
    ConvA(sim->host->QKA1int, sim->host->SigmaRB1int, sim->host->t1grid.back()) - (qK * sim->host->rInt);
    return d1qK + (d2qK * sim->host->theta);
}

void replaceAll(const vector<double>& qK,
                const vector<double>& qR,
                const vector<double>& dqK,
                const vector<double>& dqR,
                const double dr,
                const double t)
{
    // Replace the existing values in the vectors with the new values
    size_t replaceLength = qK.size();
    size_t length = sim->host->QKv.size() - replaceLength;
    if (replaceLength != qR.size() || replaceLength != dqK.size() || replaceLength != dqR.size()) {
        throw invalid_argument("All input vectors must have the same size.");
    }
    {
        sim->host->t1grid.back() = t;
        double tdiff = (sim->host->t1grid[sim->host->t1grid.size() - 1] - sim->host->t1grid[sim->host->t1grid.size() - 2]);

        if (sim->host->t1grid.size() > 2) {
            sim->host->delta_t_ratio.back() = tdiff /
                (sim->host->t1grid[sim->host->t1grid.size() - 2] - sim->host->t1grid[sim->host->t1grid.size() - 3]);
        }
        else {
            sim->host->delta_t_ratio.back() = 0.0;
        }

        #pragma omp parallel for
        for (size_t i = 0; i < replaceLength; i++)
        {
            sim->host->QKv[length + i] = qK[i];
            sim->host->QRv[length + i] = qR[i];
            sim->host->dQKv[length + i] = dqK[i] * tdiff;
            sim->host->dQRv[length + i] = dqR[i] * tdiff;
        }

        sim->host->drvec.back() = tdiff * dr;
        sim->host->rvec.back() = rstep();
    }
}

double rstep()
{
    vector<double> sigmaK(config.len, 0.0), sigmaR(config.len, 0.0);
    vector<double> qK = getLastLenEntries(sim->host->QKv, config.len);
    vector<double> qR = getLastLenEntries(sim->host->QRv, config.len);
    const double t = sim->host->t1grid.back();
    SigmaK(qK, sigmaK);
    SigmaR(qK, qR, sigmaR);
    return config.Gamma + ConvA(sigmaR, qK, t)[0] + ConvA(sigmaK, qR, t)[0] + sigmaK[0] * qK[0] / config.T0;
}

vector<double> QRstep(const vector<double>& qR)
{
    vector<double> d1qR = (qR * (-sim->host->rInt.back())) + ConvR(sim->host->SigmaRA2int, sim->host->QRB2int, sim->host->t1grid.back());
    vector<double> d2qR = (qR * sim->host->rInt) - ConvR(sim->host->QRA2int, sim->host->SigmaRB2int, sim->host->t1grid.back());
    return d1qR + (d2qR * sim->host->theta);
}

double drstep()
{
    vector<double> sigmaK(config.len, 0.0), sigmaR(config.len, 0.0), dsigmaK(config.len, 0.0), dsigmaR(config.len, 0.0);
    vector<double> qK = getLastLenEntries(sim->host->QKv, config.len);
    vector<double> qR = getLastLenEntries(sim->host->QRv, config.len);
    vector<double> dqK = QKstep(qK, qR);
    vector<double> dqR = QRstep(qR);
    const double t = sim->host->t1grid.back();
    SigmaK(qK, sigmaK);
    SigmaR(qK, qR, sigmaR);
    dsigmaK = (SigmaK10(qK) * dqK) + (SigmaK01(qK) * dqR);
    dsigmaR = (SigmaR10(qK, qR) * dqK) + (SigmaR01(qK, qR) * dqR);
    return ConvA(sigmaR, qK, 1)[0] + ConvA(sigmaK, qR, 1)[0] + ConvA(dsigmaR, qK, t)[0] + ConvA(dsigmaK, qR, t)[0] + ConvA(sigmaR, dqK, t)[0] + ConvA(sigmaK, dqR, t)[0] + (dsigmaK[0] * qK[0] + sigmaK[0] * dqK[0]) / config.T0;
}

double drstep2(const vector<double>& qK, const vector<double>& qR, const vector<double>& dqK, const vector<double>& dqR, const double t)
{
    vector<double> sigmaK(qK.size(), 0.0), sigmaR(qK.size(), 0.0), dsigmaK(qK.size(), 0.0), dsigmaR(qK.size(), 0.0);
    SigmaK(qK, sigmaK);
    SigmaR(qK, qR, sigmaR);
    dsigmaK = (SigmaK10(qK) * dqK) + (SigmaK01(qK) * dqR);
    dsigmaR = (SigmaR10(qK, qR) * dqK) + (SigmaR01(qK, qR) * dqR);
    return ConvA(sigmaR, qK, 1)[0] + ConvA(sigmaK, qR, 1)[0] + ConvA(dsigmaR, qK, t)[0] + ConvA(dsigmaK, qR, t)[0] + ConvA(sigmaR, dqK, t)[0] + ConvA(sigmaK, dqR, t)[0] + (dsigmaK[0] * qK[0] + sigmaK[0] * dqK[0]) / config.T0;
}

void appendAll(const vector<double>& qK,
    const vector<double>& qR,
    const vector<double>& dqK,
    const vector<double>& dqR,
    const double dr,
    const double t)
{
    size_t length = qK.size();
    if (length != qR.size() || length != dqK.size() || length != dqR.size()) {
        throw invalid_argument("All input vectors must have the same size.");
    }

    // 1) update t1grid and delta_t_ratio
    sim->host->t1grid.push_back(t);
    size_t idx = sim->host->t1grid.size() - 1;
    double tdiff = sim->host->t1grid[idx] - sim->host->t1grid[idx - 1];
    if (idx > 1) {
        double prev = sim->host->t1grid[idx - 1] - sim->host->t1grid[idx - 2];
        sim->host->delta_t_ratio.push_back(tdiff / prev);
    }
    else {
        sim->host->delta_t_ratio.push_back(0.0);
    }

    for (size_t i = 0; i < length; i++)
    {
        sim->host->QKv.push_back(qK[i]);
        sim->host->QRv.push_back(qR[i]);
        sim->host->dQKv.push_back(dqK[i] * tdiff);
        sim->host->dQRv.push_back(dqR[i] * tdiff);
    }

    // 2) finally update drvec and rvec
    sim->host->drvec.push_back(tdiff * dr);
    sim->host->rvec.push_back(rstep());
}

double energyCPU()
{
    double qk0 = sim->host->QKv.back();
    double t = sim->host->t1grid.back();

    std::vector<double> temp(config.len);

    SigmaK(getLastLenEntries(sim->host->QKv, config.len), temp);

    return -(ConvA(temp,
                   getLastLenEntries(sim->host->QRv, config.len),
                   t)[0]
             + flambda(qk0)/config.T0);
}