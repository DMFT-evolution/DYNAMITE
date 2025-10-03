---
title: src/EOMs/runge_kutta.cpp

---

# src/EOMs/runge_kutta.cpp



## Functions

|                | Name           |
| -------------- | -------------- |
| double | **[SSPRK104](#function-ssprk104)**() |
| double | **[RK54](#function-rk54)**() |
| double | **[update](#function-update)**() |

## Attributes

|                | Name           |
| -------------- | -------------- |
| SimulationConfig | **[config](#variable-config)**  |
| SimulationData * | **[sim](#variable-sim)**  |
| RKData * | **[rk](#variable-rk)**  |


## Functions Documentation

### function SSPRK104

```cpp
double SSPRK104()
```


### function RK54

```cpp
double RK54()
```


### function update

```cpp
double update()
```



## Attributes Documentation

### variable config

```cpp
SimulationConfig config;
```


### variable sim

```cpp
SimulationData * sim;
```


### variable rk

```cpp
RKData * rk;
```



## Source code

```cpp
#include "EOMs/runge_kutta.hpp"
#include "core/globals.hpp"
#include "core/config.hpp"
#include "core/config_build.hpp"
#include "EOMs/time_steps.hpp"
#include "interpolation/interpolation_core.hpp"
#include "core/vector_utils.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// External declarations for global variables
extern SimulationConfig config;
extern SimulationData* sim;
extern RKData* rk;

// CPU Runge-Kutta Methods (extracted from runge_kutta.cu)
double SSPRK104()
{
    const size_t stages = 10;
    const double amat[stages][stages] = {
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 6, 1.0 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 6, 1.0 / 6, 1.0 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 6, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 6, 1.0 / 6, 0.0, 0.0, 0.0},
        {1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 6, 1.0 / 6, 1.0 / 6, 0.0, 0.0},
        {1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 0.0}
    };
    const double bvec[stages] = { 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10 };
    const double b2vec[stages] = { 0., 2.0 / 9, 0, 0, 5.0 / 18, 1.0 / 3, 0., 0., 0., 1.0 / 6 };

    // Initialize variables
    std::vector<std::vector<double>> gKvec(stages + 1, std::vector<double>(config.len, 0.0));
    gKvec[0] = getLastLenEntries(sim->h_QKv, config.len);
    std::vector<std::vector<double>> gRvec(stages + 1, std::vector<double>(config.len, 0.0));
    gRvec[0] = getLastLenEntries(sim->h_QRv, config.len);
    std::vector<double> gtvec(stages + 1, 0.0);
    gtvec[0] = sim->h_t1grid.back();

    std::vector<double> gKe(config.len, 0.0);
    std::vector<double> gRe(config.len, 0.0);
    double gte = 0.0;

    std::vector<std::vector<double>> hKvec(stages, std::vector<double>(config.len, 0.0));
    std::vector<std::vector<double>> hRvec(stages, std::vector<double>(config.len, 0.0));
    std::vector<double> htvec(stages, 0.0);

    std::vector<std::vector<double>> posB1xvec(3, std::vector<double>(config.len, 0.0));
    std::vector<std::vector<double>> posB2xvec(3, std::vector<double>(config.len * config.len, 0.0));
    double dr = 0.0;

    // Loop over stages
    for (size_t n = 0; n < stages; ++n) {
        // Interpolation
        if (sim->h_QKv.size() == config.len || n != 0) {
            interpolate(
                (n == 0 ? std::vector<double>{} : (n == 5 ? posB1xvec[0] : (n == 6 ? posB1xvec[1] : (n == 7 ? posB1xvec[2] : sim->h_posB1xOld)))),
                (n == 0 ? std::vector<double>{} : (n == 5 ? posB2xvec[0] : (n == 6 ? posB2xvec[1] : (n == 7 ? posB2xvec[2] : sim->h_posB2xOld)))),
                (n == 5 || n == 6 || n == 7)
            );
        }

        // Update position vectors
        if (n == 2) {
            posB1xvec[0] = sim->h_posB1xOld;
            posB2xvec[0] = sim->h_posB2xOld;
        }
        else if (n == 3) {
            posB1xvec[1] = sim->h_posB1xOld;
            posB2xvec[1] = sim->h_posB2xOld;
        }
        else if (n == 4) {
            posB1xvec[2] = sim->h_posB1xOld;
            posB2xvec[2] = sim->h_posB2xOld;
        }

        hKvec[n] = QKstep();
        hRvec[n] = QRstep();
        htvec[n] = 1.0;

        // Update g and dr
        if (n == 0) {
            std::vector<double> lastQKv = getLastLenEntries(sim->h_QKv, config.len);
            std::vector<double> lastQRv = getLastLenEntries(sim->h_QRv, config.len);
            dr = drstep2(lastQKv, lastQRv, hKvec[n], hRvec[n], sim->h_t1grid.back());
            gKvec[n + 1] = gKvec[0] + hKvec[0] * (config.delta_t * amat[1][0]);
            gRvec[n + 1] = gRvec[0] + hRvec[0] * (config.delta_t * amat[1][0]);
            gtvec[n + 1] = gtvec[0] + config.delta_t * amat[1][0] * htvec[0];
            appendAll(gKvec[n + 1], gRvec[n + 1], hKvec[0], hRvec[0], htvec[0] * dr, gtvec[n + 1]);
        }
        else if (n == stages - 1) {
            gKvec[n + 1] = gKvec[0];
            gRvec[n + 1] = gRvec[0];
            gtvec[n + 1] = gtvec[0];
            for (size_t j = 0; j < stages; ++j) {
                gKvec[n + 1] += hKvec[j] * (config.delta_t * bvec[j]);
                gRvec[n + 1] += hRvec[j] * (config.delta_t * bvec[j]);
                gtvec[n + 1] += config.delta_t * bvec[j] * htvec[j];
            }
            replaceAll(gKvec[n + 1], gRvec[n + 1], hKvec[0], hRvec[0], htvec[0] * dr, gtvec[n + 1]);
        }
        else {
            gKvec[n + 1] = gKvec[0];
            gRvec[n + 1] = gRvec[0];
            gtvec[n + 1] = gtvec[0];
            for (size_t j = 0; j < n + 1; ++j) {
                gKvec[n + 1] += hKvec[j] * (config.delta_t * amat[n + 1][j]);
                gRvec[n + 1] += hRvec[j] * (config.delta_t * amat[n + 1][j]);
                gtvec[n + 1] += config.delta_t * amat[n + 1][j] * htvec[j];
            }
            replaceAll(gKvec[n + 1], gRvec[n + 1], hKvec[0], hRvec[0], htvec[0] * dr, gtvec[n + 1]);
        }
    }

    // Final interpolation
    interpolate(sim->h_posB1xOld, sim->h_posB2xOld);

    // Compute ge
    gKe = gKvec[0];
    gRe = gRvec[0];
    gte = gtvec[0];
    for (size_t j = 0; j < stages; ++j) {
        gKe += hKvec[j] * (config.delta_t * b2vec[j]);
        gRe += hRvec[j] * (config.delta_t * b2vec[j]);
        gte += config.delta_t * b2vec[j] * htvec[j];
    }

    // Compute error estimate
    double error = 0.0;
    for (size_t i = 0; i < gKvec[stages].size(); ++i) {
        error += std::abs(gKvec[stages][i] - gKe[i]);
    }
    for (size_t i = 0; i < gRvec[stages].size(); ++i) {
        error += std::abs(gRvec[stages][i] - gRe[i]);
    }
    error += std::abs(gtvec[stages] - gte);

    return error;
}

double RK54()
{
    const size_t stages = 7;
    const double amat[stages][stages] = {
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.0 / 40, 9.0 / 40, 0.0, 0.0, 0.0, 0.0, 0.0},
        {44.0 / 45, -56.0 / 15, 32.0 / 9, 0.0, 0.0, 0.0, 0.0},
        {19372.0/6561, -25360.0/2187, 64448.0/6561, -212.0/729, 0.0, 0.0, 0.0},
        {9017.0/3168, -355.0/33, 46732.0/5247, 49.0/176, -5103.0/18656, 0.0, 0.0},
        {35.0/384, 0.0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84, 0.0}
    };
    const double bvec[stages] = { 35.0/384, 0.0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84, 0.0 };
    const double b2vec[stages] = { 5179.0/57600, 0.0, 7571.0/16695, 393.0/640, -92097.0/339200, 187.0/2100, 1.0/40 };

    // Initialize variables
    std::vector<std::vector<double>> gKvec(stages + 1, std::vector<double>(config.len, 0.0));
    gKvec[0] = getLastLenEntries(sim->h_QKv, config.len);
    std::vector<std::vector<double>> gRvec(stages + 1, std::vector<double>(config.len, 0.0));
    gRvec[0] = getLastLenEntries(sim->h_QRv, config.len);
    std::vector<double> gtvec(stages + 1, 0.0);
    gtvec[0] = sim->h_t1grid.back();

    std::vector<double> gKe(config.len, 0.0);
    std::vector<double> gRe(config.len, 0.0);
    double gte = 0.0;

    std::vector<std::vector<double>> hKvec(stages, std::vector<double>(config.len, 0.0));
    std::vector<std::vector<double>> hRvec(stages, std::vector<double>(config.len, 0.0));
    std::vector<double> htvec(stages, 0.0);

    double dr = 0.0;

    // Loop over stages
    for (size_t n = 0; n < stages; ++n) {
        // Interpolation
        if (sim->h_QKv.size() == config.len || n != 0) {
            interpolate(
                (n == 0 ? std::vector<double>{} : sim->h_posB1xOld),
                (n == 0 ? std::vector<double>{} : sim->h_posB2xOld),
                false
            );
        }

        hKvec[n] = QKstep();
        hRvec[n] = QRstep();
        htvec[n] = 1.0;

        // Update g and dr
        if (n == 0) {
            std::vector<double> lastQKv = getLastLenEntries(sim->h_QKv, config.len);
            std::vector<double> lastQRv = getLastLenEntries(sim->h_QRv, config.len);
            dr = drstep2(lastQKv, lastQRv, hKvec[n], hRvec[n], sim->h_t1grid.back());
            gKvec[n + 1] = gKvec[0] + hKvec[0] * (config.delta_t * amat[1][0]);
            gRvec[n + 1] = gRvec[0] + hRvec[0] * (config.delta_t * amat[1][0]);
            gtvec[n + 1] = gtvec[0] + config.delta_t * amat[1][0] * htvec[0];
            appendAll(gKvec[n + 1], gRvec[n + 1], hKvec[0], hRvec[0], htvec[0] * dr, gtvec[n + 1]);
        }
        else if (n == stages - 1) {
            gKvec[n + 1] = gKvec[0];
            gRvec[n + 1] = gRvec[0];
            gtvec[n + 1] = gtvec[0];
            for (size_t j = 0; j < stages; ++j) {
                gKvec[n + 1] += hKvec[j] * (config.delta_t * bvec[j]);
                gRvec[n + 1] += hRvec[j] * (config.delta_t * bvec[j]);
                gtvec[n + 1] += config.delta_t * bvec[j] * htvec[j];
            }
            replaceAll(gKvec[n + 1], gRvec[n + 1], hKvec[0], hRvec[0], htvec[0] * dr, gtvec[n + 1]);
        }
        else {
            gKvec[n + 1] = gKvec[0];
            gRvec[n + 1] = gRvec[0];
            gtvec[n + 1] = gtvec[0];
            for (size_t j = 0; j < n + 1; ++j) {
                gKvec[n + 1] += hKvec[j] * (config.delta_t * amat[n + 1][j]);
                gRvec[n + 1] += hRvec[j] * (config.delta_t * amat[n + 1][j]);
                gtvec[n + 1] += config.delta_t * amat[n + 1][j] * htvec[j];
            }
            replaceAll(gKvec[n + 1], gRvec[n + 1], hKvec[0], hRvec[0], htvec[0] * dr, gtvec[n + 1]);
        }
    }

    // Final interpolation
    interpolate(sim->h_posB1xOld, sim->h_posB2xOld, true);

    // Compute ge
    gKe = gKvec[0];
    gRe = gRvec[0];
    gte = gtvec[0];
    for (size_t j = 0; j < stages; ++j) {
        gKe += hKvec[j] * (config.delta_t * b2vec[j]);
        gRe += hRvec[j] * (config.delta_t * b2vec[j]);
        gte += config.delta_t * b2vec[j] * htvec[j];
    }

    // Compute error estimate
    double error = 0.0;
    for (size_t i = 0; i < gKvec[stages].size(); ++i) {
        error += std::abs(gKvec[stages][i] - gKe[i]);
    }
    for (size_t i = 0; i < gRvec[stages].size(); ++i) {
        error += std::abs(gRvec[stages][i] - gRe[i]);
    }
    error += std::abs(gtvec[stages] - gte);

    return error;
}

double update() {
    // Call appropriate RK method based on rk->init
    if (rk->init == 1) {
        return RK54();
    } else {
        return SSPRK104();
    }
}
```


-------------------------------

Updated on 2025-10-03 at 23:06:51 +0200
