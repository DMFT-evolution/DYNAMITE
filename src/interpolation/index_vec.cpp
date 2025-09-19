#include "index_vec.hpp"
#include "globals.hpp"
#include "math_ops.hpp"
#include <vector>
#include <omp.h>
#include <numeric>

void indexVecLN3(const std::vector<double>& weights, const std::vector<size_t>& inds,
                 std::vector<double>& qk_result, std::vector<double>& qr_result, size_t len) {
    size_t prod = inds.size();
    size_t length = sim->h_QKv.size() - len;
    size_t depth = weights.size() / prod;
    const double* QK_start = &sim->h_QKv[length];
    const double* QR_start = &sim->h_QRv[length];

    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < prod; j++) {
        const double* weights_start = &weights[depth * j];
        qk_result[j] = std::inner_product(weights_start, weights_start + depth, QK_start + inds[j], 0.0);
        qr_result[j] = std::inner_product(weights_start, weights_start + depth, QR_start + inds[j], 0.0);
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
                qK_result[j + dims[1] * i] = (1 - 3 * in3 - 2 * in4) * sim->h_QKv[(inds[i] - 1) * dims[1] + j] + (3 * in3 + 2 * in4) * sim->h_QKv[inds[i] * dims[1] + j] - (weights[i] + 2 * in3 + in4) * sim->h_dQKv[inds[i] * dims[1] + j] - (in3 + in4) * sim->h_dQKv[(inds[i] + 1) * dims[1] + j] / dtratio[inds[i] + 1];
                qR_result[j + dims[1] * i] = (1 - 3 * in3 - 2 * in4) * sim->h_QRv[(inds[i] - 1) * dims[1] + j] + (3 * in3 + 2 * in4) * sim->h_QRv[inds[i] * dims[1] + j] - (weights[i] + 2 * in3 + in4) * sim->h_dQRv[inds[i] * dims[1] + j] - (in3 + in4) * sim->h_dQRv[(inds[i] + 1) * dims[1] + j] / dtratio[inds[i] + 1];
            }
        }
        else
        {
            for (size_t j = 0; j < dims[1]; j++)
            {
                qK_result[j + dims[1] * i] = (1 - in3) * sim->h_QKv[(inds[i] - 1) * dims[1] + j] - (weights[i] + in3) * sim->h_dQKv[inds[i] * dims[1] + j] + in3 * sim->h_QKv[inds[i] * dims[1] + j];
                qR_result[j + dims[1] * i] = (1 - in3) * sim->h_QRv[(inds[i] - 1) * dims[1] + j] - (weights[i] + in3) * sim->h_dQRv[inds[i] * dims[1] + j] + in3 * sim->h_QRv[inds[i] * dims[1] + j];
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
