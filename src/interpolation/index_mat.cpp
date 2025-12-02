// Ensure C/C++ math functions available across libstdc++/libc++
#include <cmath>
#include <math.h>
#include "interpolation/index_mat.hpp"
#include "core/config.hpp"
#include "simulation/simulation_data.hpp"
#include <algorithm>
#include <numeric>
#include <omp.h>

using namespace std;

// External declarations for global variables
extern SimulationConfig config;
extern SimulationData* sim;

// CPU version of indexMatAll
void indexMatAll(const vector<double>& posx, const vector<size_t>& indsy,
    const vector<double>& weightsy, const vector<double>& dtratio,
    vector<double>& qK_result, vector<double>& qR_result)
{
    size_t prod = indsy.size();
    size_t dims2 = weightsy.size();
    size_t depth = dims2 / prod;
    size_t t1len = dtratio.size();

    double inx, inx2, inx3;
    size_t inds, indsx;

    #pragma omp parallel for private(inx, inx2, inx3, indsx, inds) schedule(static)
    for (size_t j = 0; j < prod; j++)
    {
        indsx = max(min((size_t)posx[j], (size_t)(posx[prod - 1] - 0.5)), (size_t)1);
        inx = posx[j] - indsx;
        inx2 = inx * inx;
        inx3 = inx2 * inx;
        inds = (indsx - 1) * config.len + indsy[j];

        auto weights_start = weightsy.begin() + depth * j;
    if (indsx < t1len - 1)
        {
            // QK always linear
            qK_result[j] = (1 - 3 * inx2 + 2 * inx3) * inner_product(weights_start, weights_start + depth, sim->h_QKv.begin() + inds, 0.0) + (inx - 2 * inx2 + inx3) * inner_product(weights_start, weights_start + depth, sim->h_dQKv.begin() + config.len + inds, 0.0) + (3 * inx2 - 2 * inx3) * inner_product(weights_start, weights_start + depth, sim->h_QKv.begin() + config.len + inds, 0.0) + (-inx2 + inx3) * inner_product(weights_start, weights_start + depth, sim->h_dQKv.begin() + 2 * config.len + inds, 0.0) / dtratio[indsx + 1];
        if (!::config.log_response_interp) {
                qR_result[j] = (1 - 3 * inx2 + 2 * inx3) * inner_product(weights_start, weights_start + depth, sim->h_QRv.begin() + inds, 0.0) + (inx - 2 * inx2 + inx3) * inner_product(weights_start, weights_start + depth, sim->h_dQRv.begin() + config.len + inds, 0.0) + (3 * inx2 - 2 * inx3) * inner_product(weights_start, weights_start + depth, sim->h_QRv.begin() + config.len + inds, 0.0) + (-inx2 + inx3) * inner_product(weights_start, weights_start + depth, sim->h_dQRv.begin() + 2 * config.len + inds, 0.0) / dtratio[indsx + 1];
            } else {
                // Log-space with staggered derivative: derivative at left node stored at next index
                double QR_base = inner_product(weights_start, weights_start + depth, sim->h_QRv.begin() + inds, 0.0);
                double QR_curr = inner_product(weights_start, weights_start + depth, sim->h_QRv.begin() + config.len + inds, 0.0);
                double dQR_left  = inner_product(weights_start, weights_start + depth, sim->h_dQRv.begin() + config.len + inds, 0.0);         // pairs with QR_base
                double dQR_right = inner_product(weights_start, weights_start + depth, sim->h_dQRv.begin() + 2 * config.len + inds, 0.0);   // pairs with QR_curr
                if (QR_base > 0.0 && QR_curr > 0.0) {
                    double f_base = log(QR_base);
                    double f_curr = log(QR_curr);
                    double g_left  = dQR_left  / QR_base;
                    double g_right = dQR_right / QR_curr;
                    double denom = dtratio[indsx + 1];
                    double coeff1 = 1 - 3 * inx2 + 2 * inx3;         // f_base
                    double coeff2 = inx - 2 * inx2 + inx3;            // g_left
                    double coeff3 = 3 * inx2 - 2 * inx3;              // f_curr
                    double coeff4 = (-inx2 + inx3) / denom;           // g_right
                    double f_interp = coeff1 * f_base + coeff2 * g_left + coeff3 * f_curr + coeff4 * g_right;
                    qR_result[j] = exp(f_interp);
                } else {
                    qR_result[j] = (1 - 3 * inx2 + 2 * inx3) * QR_base + (inx - 2 * inx2 + inx3) * dQR_left + (3 * inx2 - 2 * inx3) * QR_curr + (-inx2 + inx3) * dQR_right / dtratio[indsx + 1];
                }
            }
        }
        else
        {
            qK_result[j] = (1 - inx2) * inner_product(weights_start, weights_start + depth, sim->h_QKv.begin() + inds, 0.0) + inx2 * inner_product(weights_start, weights_start + depth, sim->h_QKv.begin() + config.len + inds, 0.0) + (inx - inx2) * inner_product(weights_start, weights_start + depth, sim->h_dQKv.begin() + config.len + inds, 0.0);
        if (!::config.log_response_interp) {
                qR_result[j] = (1 - inx2) * inner_product(weights_start, weights_start + depth, sim->h_QRv.begin() + inds, 0.0) + inx2 * inner_product(weights_start, weights_start + depth, sim->h_QRv.begin() + config.len + inds, 0.0) + (inx - inx2) * inner_product(weights_start, weights_start + depth, sim->h_dQRv.begin() + config.len + inds, 0.0);
            } else {
                // Boundary segment uses only left derivative (stored at next index)
                double QR_base = inner_product(weights_start, weights_start + depth, sim->h_QRv.begin() + inds, 0.0);
                double QR_curr = inner_product(weights_start, weights_start + depth, sim->h_QRv.begin() + config.len + inds, 0.0);
                double dQR_left = inner_product(weights_start, weights_start + depth, sim->h_dQRv.begin() + config.len + inds, 0.0);
                if (QR_base > 0.0 && QR_curr > 0.0) {
                    double f_base = log(QR_base);
                    double f_curr = log(QR_curr);
                    double g_left = dQR_left / QR_base;
                    double f_interp = (1 - inx2) * f_base + inx2 * f_curr + (inx - inx2) * g_left;
                    qR_result[j] = exp(f_interp);
                } else {
                    qR_result[j] = (1 - inx2) * QR_base + inx2 * QR_curr + (inx - inx2) * dQR_left;
                }
            }
        }
    }
}
