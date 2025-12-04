#include "sparsify/sparsify_utils.hpp"
#include "simulation/simulation_data.hpp"
#include "core/config.hpp"
#include <vector>
#include <cmath>
// Prefer C math overloads to avoid namespace issues
using ::log;

extern SimulationConfig config;
extern SimulationData* sim;

// CPU-only version of sparsifyNscale
// This is a simplified implementation that performs basic sparsification
void sparsifyNscale(double threshold) {
    bool erased = false;
    std::vector<size_t> inds = {0};
    inds.reserve(sim->h_t1grid.size());

    for (size_t i = 2; i + 1 < sim->h_t1grid.size(); ++i) {
        double tleft = sim->h_t1grid[i - 2];
        double tmid  = sim->h_t1grid[i];
        double tdiff1 = sim->h_t1grid[i - 1] - tleft;
        double tdiff2 = tmid - tleft;
        double tdiff3 = sim->h_t1grid[i + 1] - tmid;

        double val = 0.0;
        for (int j = 0; j < config.len; ++j) {
            double df_term1 = sim->h_dQKv[(i - 1) * config.len + j];
            double df_term2 = sim->h_dQKv[(i + 1) * config.len + j];
            double f_term1 = sim->h_QKv[i * config.len + j] - sim->h_QKv[(i - 2) * config.len + j];
            val += std::abs(tdiff2 / 12.0 * (2 * f_term1 - tdiff2 * (df_term1 / tdiff1 + df_term2 / tdiff3)));
        }

        for (int j = 0; j < config.len; ++j) {
            // QR/dQR contribution must be measured in linear domain even when interpolation uses log(QR)
            const double QR_im2 = sim->h_QRv[(i - 2) * config.len + j];
            const double QR_i   = sim->h_QRv[i * config.len + j];
            const double dQR_im1 = sim->h_dQRv[(i - 1) * config.len + j];
            const double dQR_ip1 = sim->h_dQRv[(i + 1) * config.len + j];
            const double f_term1 = QR_i - QR_im2;
            const double df_term1 = dQR_im1;
            const double df_term2 = dQR_ip1;
            val += std::abs(tdiff2 / 12.0 * (2 * f_term1 - tdiff2 * (df_term1 / tdiff1 + df_term2 / tdiff3)));
        }

        double dRterm1 = sim->h_drvec[i - 1];
        double dRterm2 = sim->h_drvec[i + 1];
        double Rterm = sim->h_rvec[i] - sim->h_rvec[i - 2];
        val += std::abs(tdiff2 / 12.0 * (2 * Rterm - tdiff2 * (dRterm1 / tdiff1 + dRterm2 / tdiff3)));

        if (val < threshold) {
            erased = true;
        } else {
            inds.push_back(i);
        }
    }

    inds.push_back(sim->h_t1grid.size() - 1);

    if (!erased) return;

    // Rebuild the vectors by keeping only the elements at inds
    std::vector<double> new_QKv, new_QRv, new_dQKv, new_dQRv;
    std::vector<double> new_rvec, new_drvec;
    std::vector<double> new_t1grid, new_delta_t_ratio;
    
    new_QKv.reserve(inds.size() * config.len);
    new_QRv.reserve(inds.size() * config.len);
    new_dQKv.reserve(inds.size() * config.len);
    new_dQRv.reserve(inds.size() * config.len);
    new_rvec.reserve(inds.size());
    new_drvec.reserve(inds.size());
    new_t1grid.reserve(inds.size());
    new_delta_t_ratio.reserve(inds.size());

    for (size_t idx : inds) {
        for (int j = 0; j < config.len; ++j) {
            new_QKv.push_back(sim->h_QKv[idx * config.len + j]);
            new_QRv.push_back(sim->h_QRv[idx * config.len + j]);
            new_dQKv.push_back(sim->h_dQKv[idx * config.len + j]);
            new_dQRv.push_back(sim->h_dQRv[idx * config.len + j]);
        }
        new_rvec.push_back(sim->h_rvec[idx]);
        new_drvec.push_back(sim->h_drvec[idx]);
        new_t1grid.push_back(sim->h_t1grid[idx]);
        new_delta_t_ratio.push_back(sim->h_delta_t_ratio[idx]);
    }

    sim->h_QKv = std::move(new_QKv);
    sim->h_QRv = std::move(new_QRv);
    sim->h_dQKv = std::move(new_dQKv);
    sim->h_dQRv = std::move(new_dQRv);
    sim->h_rvec = std::move(new_rvec);
    sim->h_drvec = std::move(new_drvec);
    sim->h_t1grid = std::move(new_t1grid);
    sim->h_delta_t_ratio = std::move(new_delta_t_ratio);
}
