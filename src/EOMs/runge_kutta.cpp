#include "EOMs/runge_kutta.hpp"
#include "core/globals.hpp"

#include "core/config.hpp"
#include "core/config_build.hpp"
#include "EOMs/time_steps.hpp"
#include "interpolation/interpolation_core.hpp"
#include "core/vector_utils.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

// External declarations for global variables
extern SimulationConfig config;
extern SimulationData* sim;
extern RKData* rk;

// [CPC-REV] Error norm used by the adaptive controller (referee 2, point 2).
//
// Two changes with respect to the first release:
//  (a) the norm is now selectable (config.error_norm, CLI --error-norm), because
//      the historical default -- the *unnormalized* 1-norm of the 2L-component
//      error vector -- is extensive in L, so a fixed tolerance means a different
//      pointwise accuracy for L = 512, 1024, 2048.  'sum' reproduces the old
//      behaviour bit for bit and remains the default;
//  (b) the term |gt - gte| that used to be added to the error has been removed.
//      Since sum_j b_j = sum_j b2_j = 1 exactly for both embedded pairs, that
//      term is analytically zero; on the CPU it was accumulated separately and
//      therefore contributed pure rounding noise ~ulp(t), which reaches ~2e-8 at
//      t = 1e7 -- two orders of magnitude above the default tolerance 1e-10, and
//      would eventually take over the step-size controller at late times.
static double rk_error_norm(const std::vector<double>& C1, const std::vector<double>& C2,
                            const std::vector<double>& R1, const std::vector<double>& R2)
{
    const size_t n = C1.size();
    double s1 = 0.0, s2 = 0.0, smax = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double dc = std::abs(C1[i] - C2[i]);
        const double dr = std::abs(R1[i] - R2[i]);
        s1 += dc + dr;
        s2 += dc * dc + dr * dr;
        smax = std::max(smax, std::max(dc, dr));
    }
    const double N = static_cast<double>(2 * n);
    switch (config.error_norm) {
        case 1:  return s1 / N;               // mean absolute
        case 2:  return std::sqrt(s2 / N);    // root mean square
        case 3:  return smax;                 // maximum
        default: return s1;                   // extensive 1-norm (legacy default)
    }
}

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
    gKvec[0] = getLastLenEntries(sim->host->QKv, config.len);
    std::vector<std::vector<double>> gRvec(stages + 1, std::vector<double>(config.len, 0.0));
    gRvec[0] = getLastLenEntries(sim->host->QRv, config.len);
    std::vector<double> gtvec(stages + 1, 0.0);
    gtvec[0] = sim->host->t1grid.back();

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
        if (sim->host->QKv.size() == config.len || n != 0) {
            interpolate(
                (n == 0 ? std::vector<double>{} : (n == 5 ? posB1xvec[0] : (n == 6 ? posB1xvec[1] : (n == 7 ? posB1xvec[2] : sim->host->posB1xOld)))),
                (n == 0 ? std::vector<double>{} : (n == 5 ? posB2xvec[0] : (n == 6 ? posB2xvec[1] : (n == 7 ? posB2xvec[2] : sim->host->posB2xOld)))),
                (n == 5 || n == 6 || n == 7)
            );
        }

        // Update position vectors
        if (n == 2) {
            posB1xvec[0] = sim->host->posB1xOld;
            posB2xvec[0] = sim->host->posB2xOld;
        }
        else if (n == 3) {
            posB1xvec[1] = sim->host->posB1xOld;
            posB2xvec[1] = sim->host->posB2xOld;
        }
        else if (n == 4) {
            posB1xvec[2] = sim->host->posB1xOld;
            posB2xvec[2] = sim->host->posB2xOld;
        }

        hKvec[n] = QKstep(gKvec[n], gRvec[n]);
        hRvec[n] = QRstep(gRvec[n]);
        htvec[n] = 1.0;

        // Update g and dr
        if (n == 0) {
            std::vector<double> lastQKv = getLastLenEntries(sim->host->QKv, config.len);
            std::vector<double> lastQRv = getLastLenEntries(sim->host->QRv, config.len);
            dr = drstep2(lastQKv, lastQRv, hKvec[n], hRvec[n], sim->host->t1grid.back());
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
    interpolate(sim->host->posB1xOld, sim->host->posB2xOld);

    // Compute ge
    gKe = gKvec[0];
    gRe = gRvec[0];
    gte = gtvec[0];
    for (size_t j = 0; j < stages; ++j) {
        gKe += hKvec[j] * (config.delta_t * b2vec[j]);
        gRe += hRvec[j] * (config.delta_t * b2vec[j]);
        gte += config.delta_t * b2vec[j] * htvec[j];
    }

    // [CPC-REV] Eq. (25) of the manuscript, with a selectable norm and without the
    // identically-zero (but numerically noisy) time component.  See rk_error_norm().
    (void)gte;   // kept for readability of the embedded-pair assembly above
    return rk_error_norm(gKvec[stages], gKe, gRvec[stages], gRe);
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
    gKvec[0] = getLastLenEntries(sim->host->QKv, config.len);
    std::vector<std::vector<double>> gRvec(stages + 1, std::vector<double>(config.len, 0.0));
    gRvec[0] = getLastLenEntries(sim->host->QRv, config.len);
    std::vector<double> gtvec(stages + 1, 0.0);
    gtvec[0] = sim->host->t1grid.back();

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
        if (sim->host->QKv.size() == config.len || n != 0) {
            interpolate(
                (n == 0 ? std::vector<double>{} : sim->host->posB1xOld),
                (n == 0 ? std::vector<double>{} : sim->host->posB2xOld),
                false
            );
        }

        // [CPC-REV] BUG FIX (referee 2, point 5). The stage derivatives must be
        // evaluated at the CURRENT stage state, exactly as in SSPRK104() above and
        // in RK54GPU()/SSPRK104GPU(), which pass sim->device->QKv/QRv to
        // QKRstepGPU() and there slice the last `len` entries -- i.e. gKvec[n] /
        // gRvec[n], written by replaceAll() at the end of the previous stage.
        // With gKvec[0]/gRvec[0] the pointwise part of the right-hand side
        // (-mu C, -mu R, the beta_0 forcing and the 2T R bath term) was frozen at
        // stage zero, which reduces the CPU Dormand-Prince integrator from fifth
        // to FIRST order globally, while the embedded estimator stays blind to it
        // because sum_n (b_n - b2_n) = 0 exactly.
        hKvec[n] = QKstep(gKvec[n], gRvec[n]);
        hRvec[n] = QRstep(gRvec[n]);
        htvec[n] = 1.0;

        // Update g and dr
        if (n == 0) {
            std::vector<double> lastQKv = getLastLenEntries(sim->host->QKv, config.len);
            std::vector<double> lastQRv = getLastLenEntries(sim->host->QRv, config.len);
            dr = drstep2(lastQKv, lastQRv, hKvec[n], hRvec[n], sim->host->t1grid.back());
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
    interpolate(sim->host->posB1xOld, sim->host->posB2xOld, true);

    // Compute ge
    gKe = gKvec[0];
    gRe = gRvec[0];
    gte = gtvec[0];
    for (size_t j = 0; j < stages; ++j) {
        gKe += hKvec[j] * (config.delta_t * b2vec[j]);
        gRe += hRvec[j] * (config.delta_t * b2vec[j]);
        gte += config.delta_t * b2vec[j] * htvec[j];
    }

    // [CPC-REV] Eq. (25) of the manuscript, with a selectable norm and without the
    // identically-zero (but numerically noisy) time component.  See rk_error_norm().
    (void)gte;   // kept for readability of the embedded-pair assembly above
    return rk_error_norm(gKvec[stages], gKe, gRvec[stages], gRe);
}

double updateCPU() {
    // Call appropriate RK method based on rk->host->init
    if (rk->host->init == 1) {
        return RK54();
    } else {
        return SSPRK104();
    }
}
