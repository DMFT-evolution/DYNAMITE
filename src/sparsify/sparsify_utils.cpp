#include "sparsify/sparsify_utils.hpp"
#include "simulation/simulation_data.hpp"
#include "core/config.hpp"
#include <vector>
#include <cmath>
// Prefer C math overloads to avoid namespace issues
using ::log;

extern SimulationConfig config;
extern SimulationData* sim;

// ===========================================================================
// [CPC-REV] SINGLE HOST IMPLEMENTATION OF THE SPARSIFICATION SWEEP.
//
// Until this revision the repository contained TWO different definitions of
// sparsifyNscale():
//   * src/sparsify/sparsify_utils.cu  -- compiled in CUDA builds,
//   * src/sparsify/sparsify_utils.cpp -- compiled only in CPU-only builds
//     (CMakeLists.txt appended it to CPU_SOURCES under `if(NOT DMFE_WITH_CUDA)`).
// They were not equivalent, so "the CPU backend" behaved differently depending on
// whether CUDA had been enabled at configure time.  The CPU-only version
//   (a) had no alternation guard, so two adjacent slices could be removed in the
//       same sweep (contradicting Sec. 3.6 of the manuscript);
//   (b) kept slice i while the residual is built on the window [t_{i-2}, t_i],
//       i.e. it was shifted by one index with respect to the CUDA version;
//   (c) copied the stored derivatives verbatim instead of remapping them with
//       (indsD, tfac).  Because dQKv[n] holds C'(t_{n-1}) * (t_n - t_{n-1}) -- the
//       derivative of the LEFT node, scaled by the interval -- omitting the remap
//       leaves the reduced history inconsistent with its own cubic Hermite
//       interpolant.  This is the most consequential of the three.
//   (d) added the Lagrange-multiplier channel (rvec/drvec) to the residual, which
//       the CUDA version does not.
// The definition below is the CUDA-build one verbatim, so that every published
// result is reproduced bit for bit, and it is now compiled unconditionally.
// See referee 2, points 2 and 3 (CPU/GPU reproducibility).
// ===========================================================================

// CPU gather lambda function implementation
auto gather = [](const std::vector<double>& v,
                 const std::vector<size_t>& idxs,
                 size_t len,
                 const std::vector<double>& scale = {}) {
    std::vector<double> out(idxs.size() * len);
    for (size_t i = 0; i < idxs.size(); ++i) {
        size_t offset = idxs[i] * len;
        double factor = (scale.empty() ? 1.0 : scale[i]);
        for (size_t j = 0; j < len; ++j) {
            out[i * len + j] = factor * v[offset + j];
        }
    }
    return std::move(out);
};

void sparsifyNscale(double threshold) {
    bool erased = false;
    int loop = 0;
    std::vector<size_t> inds = {0};
    inds.reserve(sim->host->t1grid.size());

    for (size_t i = 2; i + 1 < sim->host->t1grid.size(); ++i) {
        double tleft = sim->host->t1grid[i - 2];
        double tmid  = sim->host->t1grid[i];
        double tdiff1 = sim->host->t1grid[i - 1] - tleft;
        double tdiff2 = tmid - tleft;
        double tdiff3 = sim->host->t1grid[i + 1] - tmid;

        double val = 0.0;
        for (int j = 0; j < config.len; ++j) {
            double df_term1 = sim->host->dQKv[(i - 1) * config.len + j];
            double df_term2 = sim->host->dQKv[(i + 1) * config.len + j];
            double f_term1 = sim->host->QKv[i * config.len + j] - sim->host->QKv[(i - 2) * config.len + j];
            val += std::abs(tdiff2 / 12.0 * (2 * f_term1 - tdiff2 * (df_term1 / tdiff1 + df_term2 / tdiff3)));
        }

        for (int j = 0; j < config.len; ++j) {
            double df_term1 = sim->host->dQRv[(i - 1) * config.len + j];
            double df_term2 = sim->host->dQRv[(i + 1) * config.len + j];
            double f_term1 = sim->host->QRv[i * config.len + j] - sim->host->QRv[(i - 2) * config.len + j];
            val += std::abs(tdiff2 / 12.0 * (2 * f_term1 - tdiff2 * (df_term1 / tdiff1 + df_term2 / tdiff3)));
        }

        if (val < threshold && !erased) {
            erased = true;
            ++loop;
        } else {
            erased = false;
            ++loop;
            inds.push_back(loop);
        }
    }

    inds.push_back(sim->host->t1grid.size() - 2);
    inds.push_back(sim->host->t1grid.size() - 1);

    // Calculate rotated and modded indices
    std::vector<size_t> indsD(inds.size());
    indsD[0] = 0; // First index remains the same
    for (size_t i = 0; i < inds.size() - 1; ++i) {
        indsD[i + 1] = inds[i] + 1;
    }

    // Compute tfac
    std::vector<double> tfac(inds.size());
    tfac[0] = 1.0;
    for (size_t i = 1; i < inds.size(); ++i) {
        tfac[i] = (sim->host->t1grid[inds[i]] - sim->host->t1grid[inds[i - 1]]) / (sim->host->t1grid[indsD[i]] - sim->host->t1grid[indsD[i] - 1]);
    }

    // printVectorDifference(gather(rvec, inds, 1),rvec);

    sim->host->QKv   = gather(sim->host->QKv, inds, config.len);
    sim->host->QRv   = gather(sim->host->QRv, inds, config.len);
    sim->host->dQKv  = gather(sim->host->dQKv, indsD, config.len, tfac);
    sim->host->dQRv  = gather(sim->host->dQRv, indsD, config.len, tfac);
    sim->host->rvec  = gather(sim->host->rvec, inds, 1);
    sim->host->drvec = gather(sim->host->drvec, indsD, 1, tfac);
    sim->host->t1grid = gather(sim->host->t1grid, inds, 1);

    // Δt ratio
    std::vector<double> dgrid(inds.size());
    for (size_t i = 1; i < sim->host->t1grid.size(); ++i)
        dgrid[i] = sim->host->t1grid[i] - sim->host->t1grid[i - 1];
    for (size_t i = 2; i < sim->host->t1grid.size(); ++i)
        sim->host->delta_t_ratio[i] = dgrid[i] / dgrid[i - 1];

    sim->host->delta_t_ratio.resize(inds.size());

    // Note: interpolate() should be called by the caller after sparsification
}
