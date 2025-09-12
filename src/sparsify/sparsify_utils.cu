#include "sparsify_utils.hpp"
#include "config.hpp"
#include "simulation_data.hpp"
#include "host_device_utils.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/extrema.h>

// External declarations for global variables
extern SimulationConfig config;
extern SimulationData* sim;

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
            double df_term1 = sim->h_dQRv[(i - 1) * config.len + j];
            double df_term2 = sim->h_dQRv[(i + 1) * config.len + j];
            double f_term1 = sim->h_QRv[i * config.len + j] - sim->h_QRv[(i - 2) * config.len + j];
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

    inds.push_back(sim->h_t1grid.size() - 2);
    inds.push_back(sim->h_t1grid.size() - 1);

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
        tfac[i] = (sim->h_t1grid[inds[i]] - sim->h_t1grid[inds[i - 1]]) / (sim->h_t1grid[indsD[i]] - sim->h_t1grid[indsD[i] - 1]);
    }

    // printVectorDifference(gather(rvec, inds, 1),rvec);

    sim->h_QKv   = gather(sim->h_QKv, inds, config.len);
    sim->h_QRv   = gather(sim->h_QRv, inds, config.len);
    sim->h_dQKv  = gather(sim->h_dQKv, indsD, config.len, tfac);
    sim->h_dQRv  = gather(sim->h_dQRv, indsD, config.len, tfac);
    sim->h_rvec  = gather(sim->h_rvec, inds, 1);
    sim->h_drvec = gather(sim->h_drvec, indsD, 1, tfac);
    sim->h_t1grid = gather(sim->h_t1grid, inds, 1);

    // Δt ratio
    std::vector<double> dgrid(inds.size());
    for (size_t i = 1; i < sim->h_t1grid.size(); ++i)
        dgrid[i] = sim->h_t1grid[i] - sim->h_t1grid[i - 1];
    for (size_t i = 2; i < sim->h_t1grid.size(); ++i)
        sim->h_delta_t_ratio[i] = dgrid[i] / dgrid[i - 1];

    sim->h_delta_t_ratio.resize(inds.size());

    // Note: interpolate() should be called by the caller after sparsification
}

// GPU gather kernel
__global__ void gatherKernel(const double* __restrict__ v,
                            const size_t* __restrict__ idxs,
                            const double* __restrict__ scale,
                            double* __restrict__ out,
                            size_t len, size_t n_chunks, bool use_scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_chunks * len) return;

    size_t chunk = i / len;
    size_t j = i % len;

    size_t offset = idxs[chunk] * len;
    double factor = use_scale ? scale[chunk] : 1.0;
    out[i] = factor * v[offset + j];
}

thrust::device_vector<double> gatherGPU(const thrust::device_vector<double>& v,
                                        const thrust::device_vector<size_t>& idxs,
                                        size_t len,
                                        const thrust::device_vector<double>& scale,
                                        cudaStream_t stream) {
    size_t n_chunks = idxs.size();
    thrust::device_vector<double> out(n_chunks * len);

    size_t threads = 64;
    size_t blocks = (n_chunks * len + threads - 1) / threads;

    bool use_scale = !scale.empty();
    gatherKernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(v.data()),
        thrust::raw_pointer_cast(idxs.data()),
        use_scale ? thrust::raw_pointer_cast(scale.data()) : nullptr,
        thrust::raw_pointer_cast(out.data()),
        len, n_chunks, use_scale);

    return std::move(out);
}

__global__ void computeSparsifyFlags(const double* __restrict__ t1grid,
                                     const double* __restrict__ QKv,
                                     const double* __restrict__ QRv,
                                     const double* __restrict__ dQKv,
                                     const double* __restrict__ dQRv,
                                     bool* __restrict__ flags,
                                     double threshold, size_t len, size_t n,
                                     cudaStream_t stream) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (i + 1 >= n) return;
    if (i/2 * 2 != i) return; // Ensure i is even

    double tleft = t1grid[i - 2];
    double tmid = t1grid[i];
    double tdiff1 = t1grid[i - 1] - tleft;
    double tdiff2 = tmid - tleft;
    double tdiff3 = t1grid[i + 1] - tmid;
    double scale = tdiff2 / 12.0;

    double val = 0.0;
    for (size_t j = 0; j < len; ++j) {
        size_t idx_im2 = (i - 2) * len + j;
        size_t idx_im1 = (i - 1) * len + j;
        size_t idx_i   = i * len + j;
        size_t idx_ip1 = (i + 1) * len + j;

        double df1_qk = dQKv[idx_im1];
        double df2_qk = dQKv[idx_ip1];
        double f_qk = QKv[idx_i] - QKv[idx_im2];

        double df1_qr = dQRv[idx_im1];
        double df2_qr = dQRv[idx_ip1];
        double f_qr = QRv[idx_i] - QRv[idx_im2];

        val += fabs(scale * (2.0 * f_qk - tdiff2 * (df1_qk / tdiff1 + df2_qk / tdiff3)));
        val += fabs(scale * (2.0 * f_qr - tdiff2 * (df1_qr / tdiff1 + df2_qr / tdiff3)));
    }
    flags[i] = (val >= threshold);
}

void sparsifyNscaleGPU(double threshold, cudaStream_t stream) {

    size_t t1len = sim->d_t1grid.size();
    thrust::device_vector<bool> flags(t1len, true);

    size_t threads = 64;
    size_t blocks = (t1len - 2 + threads - 1) / threads;
    computeSparsifyFlags<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(sim->d_t1grid.data()),
        thrust::raw_pointer_cast(sim->d_QKv.data()),
        thrust::raw_pointer_cast(sim->d_QRv.data()),
        thrust::raw_pointer_cast(sim->d_dQKv.data()),
        thrust::raw_pointer_cast(sim->d_dQRv.data()),
        thrust::raw_pointer_cast(flags.data()),
        threshold, config.len, t1len);

    thrust::device_vector<size_t> inds(t1len);
    thrust::sequence(inds.begin(), inds.end());

    size_t n = inds.size();
    thrust::device_vector<size_t> filtered(t1len); // max possible size
    auto end_it = thrust::copy_if(
        inds.begin(), inds.end(), 
        flags.begin(), 
        filtered.begin(), 
        thrust::identity<bool>()
    );
    filtered.resize(end_it - filtered.begin()); // shrink to actual size

    // Construct d_indsD and tfac
    thrust::device_vector<size_t> indsD(filtered.size(),0);
    thrust::transform(filtered.begin(), filtered.end() - 1, indsD.begin() + 1, indsD.begin() + 1, thrust::placeholders::_1 + 1);

    thrust::device_vector<double> tfac(filtered.size(), 1.0);
    const double* t1_ptr = thrust::raw_pointer_cast(sim->d_t1grid.data());

    auto max_it = thrust::max_element(indsD.begin(), indsD.end());
    auto min_it = thrust::min_element(indsD.begin(), indsD.end());
    double max_val = *max_it;
    double min_val = *min_it;

    auto max_f = thrust::max_element(filtered.begin(), filtered.end());
    auto min_f = thrust::min_element(filtered.begin(), filtered.end());
    double max_valf = *max_f;
    double min_valf = *min_f;

    tfac[0]= 1.0;
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(
            filtered.begin() + 1,       // inds[i]
            filtered.begin(),           // inds[i-1]
            indsD.begin() + 1      // indsD[i]
        )),
        thrust::make_zip_iterator(thrust::make_tuple(
            filtered.end(),             // one past last valid index
            filtered.end() - 1,
            indsD.end()
        )),
        tfac.begin() + 1,
        [t1_ptr] __device__ (thrust::tuple<size_t, size_t, size_t> tup) {
            size_t inds_i    = thrust::get<0>(tup);
            size_t inds_im1  = thrust::get<1>(tup);
            size_t indsD_i   = thrust::get<2>(tup);
            return (t1_ptr[inds_i] - t1_ptr[inds_im1]) /
                (t1_ptr[indsD_i] - t1_ptr[indsD_i - 1]);
        });

    sim->d_QKv = gatherGPU(sim->d_QKv, filtered, config.len);
    sim->d_QRv = gatherGPU(sim->d_QRv, filtered, config.len);
    sim->d_dQKv = gatherGPU(sim->d_dQKv, indsD, config.len, tfac);
    sim->d_dQRv = gatherGPU(sim->d_dQRv, indsD, config.len, tfac);
    sim->d_rvec = gatherGPU(sim->d_rvec, filtered, 1);
    sim->d_drvec = gatherGPU(sim->d_drvec, indsD, 1, tfac);
    sim->d_t1grid = gatherGPU(sim->d_t1grid, filtered, 1);

    size_t new_n = sim->d_t1grid.size();
    sim->d_delta_t_ratio.resize(new_n);
    thrust::device_vector<double> dgrid(new_n);
    thrust::transform(sim->d_t1grid.begin() + 1, sim->d_t1grid.end(), sim->d_t1grid.begin(), dgrid.begin() + 1, thrust::minus<>());
    thrust::transform(dgrid.begin() + 2, dgrid.end(), dgrid.begin() + 1, sim->d_delta_t_ratio.begin() + 2, thrust::divides<>());

    // vector<double> gpu_result(tfac.size());
    // thrust::copy(tfac.begin(), tfac.end(), gpu_result.begin());

    // Note: interpolateGPU() should be called by the caller after sparsification
}
