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

// Prefer global C overloads to avoid namespace lookup issues
using ::exp;
using ::log;

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
        // QK: always linear domain
        qk_result[j] = std::inner_product(weights_start, weights_start + depth, QK_start + inds[j], 0.0);
        if (!::config.log_response_interp) {
            qr_result[j] = std::inner_product(weights_start, weights_start + depth, QR_start + inds[j], 0.0);
        } else {
            // Use precomputed log values; fallback to linear if any sample non-positive (stored as original value)
            double lin_sum = 0.0;
            long double log_sum = 0.0L;
            bool invalid = false;
            for (size_t d = 0; d < depth; ++d) {
                const size_t idx = inds[j] + d;
                const double val = QR_start[idx];
                const double w = weights_start[d];
                lin_sum += w * val;
                if (val > 0.0) {
                    log_sum += static_cast<long double>(w) * logQR_cache[idx];
                } else {
                    invalid = true;
                }
            }
            qr_result[j] = invalid ? lin_sum : exp(static_cast<double>(log_sum));
        }
    }
}

void indexVecN(const size_t length, const std::vector<double>& weights, const std::vector<size_t>& inds,
               const std::vector<double>& dtratio, std::vector<double>& qK_result, std::vector<double>& qR_result, size_t len)
{
    size_t dims[] = {len, len};
    size_t t1len = dtratio.size();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < dims[0]; i++)
    {
        double in3 = weights[i] * weights[i];
        double in4 = in3 * weights[i];
        if (inds[i] < t1len - 1)
        {
            for (size_t j = 0; j < dims[1]; j++)
            {
        // QK in linear domain
        qK_result[j + dims[1] * i] = (1 - 3 * in3 - 2 * in4) * sim->h_QKv[(inds[i] - 1) * dims[1] + j] + (3 * in3 + 2 * in4) * sim->h_QKv[inds[i] * dims[1] + j] - (weights[i] + 2 * in3 + in4) * sim->h_dQKv[inds[i] * dims[1] + j] - (in3 + in4) * sim->h_dQKv[(inds[i] + 1) * dims[1] + j] / dtratio[inds[i] + 1];
        if (!::config.log_response_interp) {
                    qR_result[j + dims[1] * i] = (1 - 3 * in3 - 2 * in4) * sim->h_QRv[(inds[i] - 1) * dims[1] + j] + (3 * in3 + 2 * in4) * sim->h_QRv[inds[i] * dims[1] + j] - (weights[i] + 2 * in3 + in4) * sim->h_dQRv[inds[i] * dims[1] + j] - (in3 + in4) * sim->h_dQRv[(inds[i] + 1) * dims[1] + j] / dtratio[inds[i] + 1];
                } else {
                    // Staggered storage: derivative at left node (i-1) is stored at index i,
                    // and derivative at right node (i) is stored at index i+1.
                    const size_t base_i = inds[i] - 1;
                    const size_t curr_i = inds[i];
                    const double qR_base = sim->h_QRv[base_i * dims[1] + j];
                    const double qR_curr = sim->h_QRv[curr_i * dims[1] + j];
                    const double dqR_left  = sim->h_dQRv[curr_i * dims[1] + j];     // pairs with qR_base
                    const double dqR_right = sim->h_dQRv[(curr_i + 1) * dims[1] + j]; // pairs with qR_curr
                    if (qR_base > 0.0 && qR_curr > 0.0) {
                        const double f_base = log(qR_base);
                        const double f_curr = log(qR_curr);
                        const double g_left  = dqR_left  / qR_base; // d(log QR) at left node
                        const double g_right = dqR_right / qR_curr; // d(log QR) at right node
                        const double coeff1 = 1 - 3 * in3 - 2 * in4;         // f_base
                        const double coeff2 = 3 * in3 + 2 * in4;             // f_curr
                        const double coeff3 = (weights[i] + 2 * in3 + in4);  // g_left
                        const double coeff4 = (in3 + in4) / dtratio[curr_i + 1]; // g_right
                        const double f_interp = coeff1 * f_base + coeff2 * f_curr - coeff3 * g_left - coeff4 * g_right;
                        qR_result[j + dims[1] * i] = exp(f_interp);
                    } else {
                        // Fallback: linear-domain Hermite
                        qR_result[j + dims[1] * i] = (1 - 3 * in3 - 2 * in4) * qR_base + (3 * in3 + 2 * in4) * qR_curr - (weights[i] + 2 * in3 + in4) * dqR_left - (in3 + in4) * dqR_right / dtratio[curr_i + 1];
                    }
                }
            }
        }
        else
        {
            for (size_t j = 0; j < dims[1]; j++)
            {
                qK_result[j + dims[1] * i] = (1 - in3) * sim->h_QKv[(inds[i] - 1) * dims[1] + j] - (weights[i] + in3) * sim->h_dQKv[inds[i] * dims[1] + j] + in3 * sim->h_QKv[inds[i] * dims[1] + j];
        if (!::config.log_response_interp) {
                    qR_result[j + dims[1] * i] = (1 - in3) * sim->h_QRv[(inds[i] - 1) * dims[1] + j] - (weights[i] + in3) * sim->h_dQRv[inds[i] * dims[1] + j] + in3 * sim->h_QRv[inds[i] * dims[1] + j];
                } else {
                    // Boundary segment uses only left derivative (stored at index curr_i)
                    const size_t base_i = inds[i] - 1;
                    const size_t curr_i = inds[i];
                    const double qR_base = sim->h_QRv[base_i * dims[1] + j];
                    const double qR_curr = sim->h_QRv[curr_i * dims[1] + j];
                    const double dqR_left = sim->h_dQRv[curr_i * dims[1] + j]; // pairs with qR_base
                    if (qR_base > 0.0 && qR_curr > 0.0) {
                        const double f_base = log(qR_base);
                        const double f_curr = log(qR_curr);
                        const double g_left = dqR_left / qR_base;
                        const double f_interp = (1 - in3) * f_base + in3 * f_curr - (weights[i] + in3) * g_left;
                        qR_result[j + dims[1] * i] = exp(f_interp);
                    } else {
                        qR_result[j + dims[1] * i] = (1 - in3) * qR_base - (weights[i] + in3) * dqR_left + in3 * qR_curr;
                    }
                }
            }
        }
    }
}

void indexVecR2(const std::vector<double>& in1, const std::vector<double>& in2, const std::vector<double>& in3,
                const std::vector<size_t>& inds, const std::vector<double>& dtratio, std::vector<double>& result)
{
    size_t dims = inds.size();
    size_t t1len = dtratio.size();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < dims; i++)
    {
        if (inds[i] < t1len - 1)
        {
            result[i] = (1 - 3 * pow_const<2>(in3[i]) - 2 * pow_const<3>(in3[i])) * in1[inds[i] - 1] + (3 * pow_const<2>(in3[i]) + 2 * pow_const<3>(in3[i])) * in1[inds[i]] - (in3[i] + 2 * pow_const<2>(in3[i]) + pow_const<3>(in3[i])) * in2[inds[i]] - (pow_const<2>(in3[i]) + pow_const<3>(in3[i])) * in2[inds[i] + 1] / dtratio[inds[i] + 1];
        }
        else
        {
            result[i] = (1 - pow_const<2>(in3[i])) * in1[inds[i] - 1] + pow_const<2>(in3[i]) * in1[inds[i]] - (in3[i] + pow_const<2>(in3[i])) * in2[inds[i]];
        }
    }
}
