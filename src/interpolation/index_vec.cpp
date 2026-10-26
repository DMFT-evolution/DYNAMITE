// Ensure C/C++ math functions available across libstdc++/libc++
#include <cmath>
#include <math.h>
#include "interpolation/index_vec.hpp"
#include "core/globals.hpp"

#include "core/config.hpp"
#include "math/math_ops.hpp"
#if defined(_OPENMP)
#include <omp.h>
#endif   // [CPC-REV] serial build without OpenMP
#include <numeric>

// Global simulation configuration (defined in main.cpp)
extern SimulationConfig config;

void indexVecLN3(const std::vector<double>& weights, const std::vector<size_t>& inds,
                 std::vector<double>& qk_result, std::vector<double>& qr_result, size_t len) {
    size_t prod = inds.size();
    size_t length = sim->host->QKv.size() - len;
    size_t depth = weights.size() / prod;
    const double* QK_start = &sim->host->QKv[length];
    const double* QR_start = &sim->host->QRv[length];

    // Optional precompute of log slice for last len entries (only once per call).
    // NOTE: Must NOT be thread_local because OpenMP creates per-thread instances that
    // would remain uninitialized in worker threads, causing out-of-bounds access.
    static std::vector<double> logQR_cache; // shared read-only after fill
    if (config.log_response_interp) {
        if (logQR_cache.size() != len) logQR_cache.resize(len);
        for (size_t i = 0; i < len; ++i) {
            double v = QR_start[i];
            logQR_cache[i] = (v > 0.0) ? log(v) : v; // store original if non-positive for fallback detection
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < prod; j++) {
        const double* weights_start = &weights[depth * j];
        if (!::config.log_response_interp) {
            // QK/Q R: linear domain
            qk_result[j] = std::inner_product(weights_start, weights_start + depth, QK_start + inds[j], 0.0);
            qr_result[j] = std::inner_product(weights_start, weights_start + depth, QR_start + inds[j], 0.0);
        } else {
            // Legacy behavior: QR interpolated in log-space with linear fallback; QK interpolated normally.
            qk_result[j] = std::inner_product(weights_start, weights_start + depth, QK_start + inds[j], 0.0);
            double qr_lin_sum = 0.0;
            long double qr_log_sum = 0.0L;
            bool qr_invalid = false;
            for (size_t d = 0; d < depth; ++d) {
                const size_t idx = inds[j] + d;
                const double w = weights_start[d];

                const double qrv = QR_start[idx];
                qr_lin_sum += w * qrv;
                if (qrv > 0.0) {
                    qr_log_sum += static_cast<long double>(w) * logQR_cache[idx];
                } else {
                    qr_invalid = true;
                }
            }

            qr_result[j] = qr_invalid ? qr_lin_sum : exp(static_cast<double>(qr_log_sum));
        }
    }
}

void indexVecN(const size_t length, const std::vector<double>& weights, const std::vector<size_t>& inds,
               const std::vector<double>& dtratio, std::vector<double>& qK_result, std::vector<double>& qR_result, size_t len)
{
    // Cubic Hermite interpolation in t at the times theta_i * t (the "B1" values).
    // dQKv[n], dQRv[n] hold the derivative at slice n-1 times (t_n - t_{n-1}).
    //
    // [CPC-REV] This routine now follows indexVecNKernel_optimized (index_vec.cu)
    // branch by branch.  Before, the CPU version
    //  - dropped the derivative term in the last interval, i.e. between the last
    //    stored slice and the current Runge-Kutta stage, where only the left
    //    derivative is available.  This lowered the order of the interpolation
    //    there and made CPU runs several orders of magnitude less accurate than GPU
    //    runs (pure p=2 benchmark up to t=1: |dC| 7e-9 instead of 2e-13);
    //  - used log(dQRv) instead of dQRv/QRv in the log-response branch, and wrong
    //    signs in its last-interval fallback;
    //  - read dtratio[curr_i + 1] out of bounds for the last interval.
    (void)length;
    const size_t t1len = dtratio.size();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < len; i++)
    {
        if (t1len < 2) {
            for (size_t j = 0; j < len; j++) {
                qK_result[j + len * i] = sim->host->QKv[j];
                qR_result[j + len * i] = sim->host->QRv[j];
            }
            continue;
        }
        const double weight = weights[i];
        const double weight2 = weight * weight;
        const double weight3 = weight2 * weight;
        const size_t curr_i = inds[i];
        const size_t base_i = curr_i - 1;
        const double* QK  = &sim->host->QKv[0];
        const double* QR  = &sim->host->QRv[0];
        const double* dQK = &sim->host->dQKv[0];
        const double* dQR = &sim->host->dQRv[0];

        if (curr_i < t1len - 1) {
            // Interior interval: derivatives at both ends.
            const double coeff1 = 1.0 - 3.0 * weight2 - 2.0 * weight3;
            const double coeff2 = 3.0 * weight2 + 2.0 * weight3;
            const double coeff3 = weight + 2.0 * weight2 + weight3;
            const double coeff4 = (weight2 + weight3) / dtratio[curr_i + 1];
            for (size_t j = 0; j < len; j++) {
                const size_t b = base_i * len + j, c = curr_i * len + j, n = c + len;
                qK_result[j + len * i] = coeff1 * QK[b] + coeff2 * QK[c] - coeff3 * dQK[c] - coeff4 * dQK[n];
                const double lin = coeff1 * QR[b] + coeff2 * QR[c] - coeff3 * dQR[c] - coeff4 * dQR[n];
                if (::config.log_response_interp && QR[b] > 0.0 && QR[c] > 0.0) {
                    const double f = coeff1 * log(QR[b]) + coeff2 * log(QR[c])
                                   - coeff3 * (dQR[c] / QR[b]) - coeff4 * (dQR[n] / QR[c]);
                    qR_result[j + len * i] = exp(f);
                } else {
                    qR_result[j + len * i] = lin;
                }
            }
        } else {
            // Last interval (up to the current stage): only the left derivative is known.
            const double coeff1 = 1.0 - weight2;
            const double coeff2 = weight + weight2;
            for (size_t j = 0; j < len; j++) {
                const size_t b = base_i * len + j, c = curr_i * len + j;
                qK_result[j + len * i] = coeff1 * QK[b] + weight2 * QK[c] - coeff2 * dQK[c];
                const double lin = coeff1 * QR[b] + weight2 * QR[c] - coeff2 * dQR[c];
                if (::config.log_response_interp && QR[b] > 0.0 && QR[c] > 0.0) {
                    const double f = coeff1 * log(QR[b]) + weight2 * log(QR[c]) - coeff2 * (dQR[c] / QR[b]);
                    qR_result[j + len * i] = exp(f);
                } else {
                    qR_result[j + len * i] = lin;
                }
            }
        }
    }
}

void indexVecR2(const std::vector<double>& in1, const std::vector<double>& in2, const std::vector<double>& in3,
                const std::vector<size_t>& inds, const std::vector<double>& dtratio, std::vector<double>& result)
{
    (void)in2;
    (void)dtratio;
    size_t dims = inds.size();
    size_t t1len = dtratio.size();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < dims; i++)
    {
        if(inds[i]<t1len-1)
        {
            const size_t base_i = inds[i] - 1;
            const size_t curr_i = inds[i];
            const double weight = in3[i];
            const double weight_sq = weight * weight;
            const double weight_cub = weight_sq * weight;
            result[i] = (1.0 - 3.0 * weight_sq - 2.0 * weight_cub) * in1[base_i] +
                        (3.0 * weight_sq + 2.0 * weight_cub) * in1[curr_i] -
                        (weight + 2.0 * weight_sq + weight_cub) * in2[curr_i] -
                        (weight_sq + weight_cub) * in2[curr_i + 1] / dtratio[curr_i + 1];
        }
        else
        { 
            const size_t base_i = inds[i] - 1;
            const size_t curr_i = inds[i];
            const double weight = in3[i];
            const double weight_sq = weight * weight;
            result[i] = (1.0 - weight_sq) * in1[base_i] + weight_sq * in1[curr_i] - (weight + weight_sq) * in2[curr_i];
        } 
    }
}
