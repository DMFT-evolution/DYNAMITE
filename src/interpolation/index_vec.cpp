// Ensure C/C++ math functions available across libstdc++/libc++
#include <cmath>
#include <math.h>
#include "interpolation/index_vec.hpp"
#include "core/globals.hpp"
#include "core/config.hpp"
#include "math/math_ops.hpp"
#include <vector>
#include <omp.h>
#include <numeric>

// Global simulation configuration (defined in main.cpp)
extern SimulationConfig config;

void indexVecLN3(const std::vector<double>& weights, const std::vector<size_t>& inds,
                 std::vector<double>& qk_result, std::vector<double>& qr_result, size_t len) {
    size_t prod = inds.size();
    size_t length = sim->h_QKv.size() - len;
    size_t depth = weights.size() / prod;
    const double* QK_start = &sim->h_QKv[length];
    const double* QR_start = &sim->h_QRv[length];

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
    (void)length;
    (void)dtratio;
    size_t dims[] = {len, len};
    size_t t1len = dtratio.size();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < dims[0]; i++)
    {
        (void)t1len;
        const double weight = weights[i];
        const size_t base_i = inds[i] - 1;
        const size_t curr_i = inds[i];
        for (size_t j = 0; j < dims[1]; j++)
        {
            const double qK_base = sim->h_QKv[base_i * dims[1] + j];
            const double qK_curr = sim->h_QKv[curr_i * dims[1] + j];
            qK_result[j + dims[1] * i] = (1.0 - weight) * qK_base + weight * qK_curr;

            const double qR_base = sim->h_QRv[base_i * dims[1] + j];
            const double qR_curr = sim->h_QRv[curr_i * dims[1] + j];
            if (!::config.log_response_interp) {
                qR_result[j + dims[1] * i] = (1.0 - weight) * qR_base + weight * qR_curr;
            } else {
                if (qR_base > 0.0 && qR_curr > 0.0) {
                    const double f_base = log(qR_base);
                    const double f_curr = log(qR_curr);
                    const double f_interp = (1.0 - weight) * f_base + weight * f_curr;
                    qR_result[j + dims[1] * i] = exp(f_interp);
                } else {
                    qR_result[j + dims[1] * i] = (1.0 - weight) * qR_base + weight * qR_curr;
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
        (void)t1len;
        const size_t base_i = inds[i] - 1;
        const size_t curr_i = inds[i];
        const double weight = in3[i];
        result[i] = (1.0 - weight) * in1[base_i] + weight * in1[curr_i];
    }
}
