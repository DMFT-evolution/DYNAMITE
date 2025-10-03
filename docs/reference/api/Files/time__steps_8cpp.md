---
title: src/EOMs/time_steps.cpp

---

# src/EOMs/time_steps.cpp



## Namespaces

| Name           |
| -------------- |
| **[std](Namespaces/namespacestd.md)**  |

## Functions

|                | Name           |
| -------------- | -------------- |
| vector< double > | **[getLastLenEntries](#function-getlastlenentries)**(const vector< double > & vec, size_t len) |
| vector< double > | **[QKstep](#function-qkstep)**() |
| void | **[replaceAll](#function-replaceall)**(const vector< double > & qK, const vector< double > & qR, const vector< double > & dqK, const vector< double > & dqR, const double dr, const double t) |
| double | **[rstep](#function-rstep)**() |
| vector< double > | **[QRstep](#function-qrstep)**() |
| double | **[drstep](#function-drstep)**() |
| double | **[drstep2](#function-drstep2)**(const vector< double > & qK, const vector< double > & qR, const vector< double > & dqK, const vector< double > & dqR, const double t) |
| void | **[appendAll](#function-appendall)**(const vector< double > & qK, const vector< double > & qR, const vector< double > & dqK, const vector< double > & dqR, const double dr, const double t) |

## Attributes

|                | Name           |
| -------------- | -------------- |
| SimulationConfig | **[config](#variable-config)**  |


## Functions Documentation

### function getLastLenEntries

```cpp
vector< double > getLastLenEntries(
    const vector< double > & vec,
    size_t len
)
```


### function QKstep

```cpp
vector< double > QKstep()
```


### function replaceAll

```cpp
void replaceAll(
    const vector< double > & qK,
    const vector< double > & qR,
    const vector< double > & dqK,
    const vector< double > & dqR,
    const double dr,
    const double t
)
```


### function rstep

```cpp
double rstep()
```


### function QRstep

```cpp
vector< double > QRstep()
```


### function drstep

```cpp
double drstep()
```


### function drstep2

```cpp
double drstep2(
    const vector< double > & qK,
    const vector< double > & qR,
    const vector< double > & dqK,
    const vector< double > & dqR,
    const double t
)
```


### function appendAll

```cpp
void appendAll(
    const vector< double > & qK,
    const vector< double > & qR,
    const vector< double > & dqK,
    const vector< double > & dqR,
    const double dr,
    const double t
)
```



## Attributes Documentation

### variable config

```cpp
SimulationConfig config;
```



## Source code

```cpp
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

double rstep()
{
    vector<double> sigmaK(config.len, 0.0), sigmaR(config.len, 0.0);
    vector<double> qK = getLastLenEntries(sim->h_QKv, config.len);
    vector<double> qR = getLastLenEntries(sim->h_QRv, config.len);
    const double t = sim->h_t1grid.back();
    SigmaK(qK, sigmaK);
    SigmaR(qK, qR, sigmaR);
    return config.Gamma + ConvA(sigmaR, qK, t)[0] + ConvA(sigmaK, qR, t)[0] + sigmaK[0] * qK[0] / config.T0;
}

vector<double> QRstep()
{
    vector<double> qR = getLastLenEntries(sim->h_QRv, config.len);
    vector<double> d1qR = (qR * (-sim->h_rInt.back())) + ConvR(sim->h_SigmaRA2int, sim->h_QRB2int, sim->h_t1grid.back());
    vector<double> d2qR = (qR * sim->h_rInt) - ConvR(sim->h_QRA2int, sim->h_SigmaRB2int, sim->h_t1grid.back());
    return d1qR + (d2qR * sim->h_theta);
}

double drstep()
{
    vector<double> sigmaK(config.len, 0.0), sigmaR(config.len, 0.0), dsigmaK(config.len, 0.0), dsigmaR(config.len, 0.0);
    vector<double> qK = getLastLenEntries(sim->h_QKv, config.len);
    vector<double> qR = getLastLenEntries(sim->h_QRv, config.len);
    vector<double> dqK = getLastLenEntries(sim->h_dQKv, config.len);
    vector<double> dqR = getLastLenEntries(sim->h_dQRv, config.len);
    const double t = sim->h_t1grid.back();
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
    sim->h_t1grid.push_back(t);
    size_t idx = sim->h_t1grid.size() - 1;
    double tdiff = sim->h_t1grid[idx] - sim->h_t1grid[idx - 1];
    if (idx > 1) {
        double prev = sim->h_t1grid[idx - 1] - sim->h_t1grid[idx - 2];
        sim->h_delta_t_ratio.push_back(tdiff / prev);
    }
    else {
        sim->h_delta_t_ratio.push_back(0.0);
    }

    for (size_t i = 0; i < length; i++)
    {
        sim->h_QKv.push_back(qK[i]);
        sim->h_QRv.push_back(qR[i]);
        sim->h_dQKv.push_back(tdiff * dqK[i]);
        sim->h_dQRv.push_back(tdiff * dqR[i]);
    }

    // 2) finally update drvec and rvec
    sim->h_drvec.push_back(tdiff * dr);
    sim->h_rvec.push_back(rstep());
}
```


-------------------------------

Updated on 2025-10-03 at 23:06:51 +0200
