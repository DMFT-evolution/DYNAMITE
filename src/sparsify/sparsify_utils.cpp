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
            // QR/dQR contribution must be measured in linear domain even when interpolation uses log(QR)
            const double QR_im2 = sim->host->QRv[(i - 2) * config.len + j];
            const double QR_i   = sim->host->QRv[i * config.len + j];
            const double dQR_im1 = sim->host->dQRv[(i - 1) * config.len + j];
            const double dQR_ip1 = sim->host->dQRv[(i + 1) * config.len + j];
            const double f_term1 = QR_i - QR_im2;
            const double df_term1 = dQR_im1;
            const double df_term2 = dQR_ip1;
            val += std::abs(tdiff2 / 12.0 * (2 * f_term1 - tdiff2 * (df_term1 / tdiff1 + df_term2 / tdiff3)));
        }

        double dRterm1 = sim->host->drvec[i - 1];
        double dRterm2 = sim->host->drvec[i + 1];
        double Rterm = sim->host->rvec[i] - sim->host->rvec[i - 2];
        val += std::abs(tdiff2 / 12.0 * (2 * Rterm - tdiff2 * (dRterm1 / tdiff1 + dRterm2 / tdiff3)));

        if (val < threshold) {
            erased = true;
        } else {
            inds.push_back(i);
        }
    }

    inds.push_back(sim->host->t1grid.size() - 1);

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
            new_QKv.push_back(sim->host->QKv[idx * config.len + j]);
            new_QRv.push_back(sim->host->QRv[idx * config.len + j]);
            new_dQKv.push_back(sim->host->dQKv[idx * config.len + j]);
            new_dQRv.push_back(sim->host->dQRv[idx * config.len + j]);
        }
        new_rvec.push_back(sim->host->rvec[idx]);
        new_drvec.push_back(sim->host->drvec[idx]);
        new_t1grid.push_back(sim->host->t1grid[idx]);
        new_delta_t_ratio.push_back(sim->host->delta_t_ratio[idx]);
    }

    sim->host->QKv = std::move(new_QKv);
    sim->host->QRv = std::move(new_QRv);
    sim->host->dQKv = std::move(new_dQKv);
    sim->host->dQRv = std::move(new_dQRv);
    sim->host->rvec = std::move(new_rvec);
    sim->host->drvec = std::move(new_drvec);
    sim->host->t1grid = std::move(new_t1grid);
    sim->host->delta_t_ratio = std::move(new_delta_t_ratio);
}
