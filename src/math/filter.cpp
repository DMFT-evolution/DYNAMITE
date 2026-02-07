#include "math/filter.hpp"
#include "core/globals.hpp"
#include "core/config.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

// External declarations for global variables
extern SimulationConfig config;
extern SimulationData* sim;

namespace {
inline double smoothstep(double t) {
    return t * t * (3.0 - 2.0 * t);
}

void apply_filter_cpu_old(double* vec,
                          const std::vector<double>& theta,
                          size_t len,
                          double alpha,
                          double dx_avg,
                          int taper_len)
{
    std::vector<double> tmp(len);
    std::copy(vec, vec + len, tmp.begin());

    for (size_t i = 0; i < len; ++i) {
        if (i == 0 || i + 1 >= len) {
            vec[i] = tmp[i];
            continue;
        }
        double dxm = (theta.size() >= len) ? (theta[i] - theta[i - 1]) : 1.0;
        double dxp = (theta.size() >= len) ? (theta[i + 1] - theta[i]) : 1.0;
        double denom = dxm + dxp;
        if (denom <= 0.0) denom = 1.0;

        double wL = dxp / denom;
        double wR = dxm / denom;
        double avg = wL * tmp[i - 1] + wR * tmp[i + 1];

        size_t d = std::min(i, len - 1 - i);
        double taper = 1.0;
        if (taper_len > 0 && d < static_cast<size_t>(taper_len)) {
            double t = static_cast<double>(d) / static_cast<double>(taper_len);
            taper = smoothstep(t);
        }

        double dx = 0.5 * (dxm + dxp);
        double wdx = (dx_avg > 0.0) ? (dx / dx_avg) : 1.0;
        wdx = std::max(0.25, std::min(4.0, wdx));

        double a = alpha * taper * wdx;
        if (a < 0.0) a = 0.0;
        if (a > 1.0) a = 1.0;

        vec[i] = tmp[i] + a * (avg - tmp[i]);
    }
}
} // namespace

void filter_old(double* gK, double* gR, double* hK0, double* hR0, size_t len)
{
    if (len < 3) return;
    if (config.filter_strength <= 0.0) return;
    if (sim == nullptr) return;

    double alpha = config.filter_strength;
    if (alpha < 0.0) return;
    if (alpha > 1.0) alpha = 1.0;

    double dx_avg = 1.0;
    if (sim->h_theta.size() >= len) {
        dx_avg = (sim->h_theta[len - 1] - sim->h_theta[0]);
        if (dx_avg <= 0.0) dx_avg = 1.0;
    }

    int taper_len = static_cast<int>(len / 20);
    if (taper_len < 2) taper_len = 2;
    if (taper_len > 32) taper_len = 32;

    apply_filter_cpu_old(gK, sim->h_theta, len, alpha, dx_avg, taper_len);
    apply_filter_cpu_old(gR, sim->h_theta, len, alpha, dx_avg, taper_len);
    apply_filter_cpu_old(hK0, sim->h_theta, len, alpha, dx_avg, taper_len);
    apply_filter_cpu_old(hR0, sim->h_theta, len, alpha, dx_avg, taper_len);
}

void apply_lowpass5_cpu(double* vec,
                        const std::vector<double>& theta,
                        size_t len,
                        double alpha)
{
    if (len < 5) return;

    constexpr double c0 = -0.003571;  // Hamming-windowed sinc, cutoff=3 rad, width=5
    constexpr double c1 =  0.024358;
    constexpr double c2 =  0.958426;

    std::vector<double> tmp(len);
    std::copy(vec, vec + len, tmp.begin());

    for (size_t i = 0; i < len; ++i) {
        if (i < 2 || i + 2 >= len) {
            vec[i] = tmp[i];
            continue;
        }
        double dxm = (theta.size() >= len) ? (theta[i] - theta[i - 1]) : 1.0;
        double w = alpha * dxm;
        if (w < 0.0) w = 0.0;
        if (w > 1.0) w = 1.0;

        double lpf = c0 * (tmp[i - 2] + tmp[i + 2])
                   + c1 * (tmp[i - 1] + tmp[i + 1])
                   + c2 * tmp[i];
        vec[i] = tmp[i] + w * (lpf - tmp[i]);
    }
}

namespace {
double filter_weight_from_hR0(const double* hR0, size_t len)
{
    if (len < 6) return 0.0;
    if (sim == nullptr) return 0.0;
    constexpr double c0 = -0.003571;
    constexpr double c1 =  0.024358;
    constexpr double c2 =  0.958426;
    double num = 0.0, den = 0.0;
    for (size_t i = 2; i + 2 < len; ++i) {
        double lpf = c0 * (hR0[i - 2] + hR0[i + 2])
                   + c1 * (hR0[i - 1] + hR0[i + 1])
                   + c2 * hR0[i];
        double lpfm1 = c0 * (hR0[i - 3] + hR0[i + 1])
                     + c1 * (hR0[i - 2] + hR0[i])
                     + c2 * hR0[i - 1];
        double dxm = (sim->h_theta.size() >= len) ? (sim->h_theta[i] - sim->h_theta[i - 1]) : 1.0;
        double Di = dxm * (lpf - hR0[i]);
        if (i > 2) {
            double Dm1 = dxm * (lpfm1 - hR0[i - 1]);
            num += std::abs(Di - Dm1);
            den += std::abs(Di + Dm1);
        }
    }
    if (den <= 0.0) return 0.0;
    double ratio = num / den;
    return (ratio > 1.0) ? (ratio - 1.0) : 0.0;
}

void filter_apply_cpu(double* gK, double* gR, double* hK0, double* hR0, size_t len)
{
    if (len < 5) return;
    if (config.filter_strength <= 0.0) return;
    if (sim == nullptr) return;
    if (hR0 == nullptr) return;

    double alpha = config.filter_strength;
    if (alpha < 0.0) return;

    double w_apply = filter_weight_from_hR0(hR0, len);
    if (w_apply <= 0.0) return;

    double scaled = alpha * w_apply;
    if (gK) apply_lowpass5_cpu(gK, sim->h_theta, len, scaled);
    if (gR) apply_lowpass5_cpu(gR, sim->h_theta, len, scaled);
    if (hK0) apply_lowpass5_cpu(hK0, sim->h_theta, len, scaled);
    if (hR0) apply_lowpass5_cpu(hR0, sim->h_theta, len, scaled);
}
} // namespace

void filter(double* gK, double* gR, double* hK0, double* hR0, size_t len)
{
    filter_apply_cpu(gK, gR, hK0, hR0, len);
}

void filter_dR(double* hR0, size_t len)
{
    filter_apply_cpu(nullptr, nullptr, nullptr, hR0, len);
}

void filter_dRK(double* hK0, double* hR0, size_t len)
{
    filter_apply_cpu(nullptr, nullptr, hK0, hR0, len);
}
