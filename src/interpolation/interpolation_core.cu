#include "interpolation/interpolation_core.hpp"
#include "core/config.hpp"
#include "simulation/simulation_data.hpp"
#include "search/search_utils.hpp"
#include "interpolation/index_vec.hpp"
#include "interpolation/index_mat.hpp"
#include "math/math_sigma.hpp"
#include "core/vector_utils.hpp"
#include "core/stream_pool.hpp"
#include "io/io_utils.hpp"
#include "core/device_utils.cuh"
#include <vector>
#include <cmath>
#include <algorithm>
#include <thrust/device_vector.h>

using namespace std;

// External declarations for global variables
extern SimulationConfig config;
extern SimulationData* sim;

__global__ void diffNfloorKernel(
    const double* __restrict__ posB1x,
    size_t* __restrict__ Floor,
    double* __restrict__ diff,
    size_t len)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    double pos = posB1x[i];

    // Floor with bounds check done purely inside the kernel
    size_t floored = static_cast<size_t>(floor(pos));

    // Instead of precomputing maxCeil, compute it on-the-fly from pos
    size_t maxCeil = static_cast<size_t>(ceil(pos)); // conservative per-thread local bound
    floored = max(size_t(1), min(floored, maxCeil - 1));

    Floor[i] = floored;
    diff[i] = static_cast<double>(floored) - pos;
}

void diffNfloor(
    const thrust::device_vector<double>& posB1x,
    thrust::device_vector<size_t>& Floor,
    thrust::device_vector<double>& diff,
    cudaStream_t stream)
{
    size_t len = posB1x.size();
    int threads = 64;
    int blocks = (len + threads - 1) / threads;

    diffNfloorKernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(posB1x.data()),
        thrust::raw_pointer_cast(Floor.data()),
        thrust::raw_pointer_cast(diff.data()),
        len
    );
}

void interpolateGPU(
    const double* posB1xIn,
    const double* posB2xIn,
    const bool same,
    StreamPool* pool) {

    size_t len = sim->h_theta.size();
    int threads = 64;
    int blocks = (len*len + threads - 1) / threads;

    if (!pool) pool = &getDefaultStreamPool();

    // Compute sim->d_posB1x
    bsearchPosSortedGPU(sim->d_t1grid, sim->d_theta, sim->d_posB1xOld, (*pool)[0]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("bsearchPosSortedGPU(theta)");

    // Compute sim->d_posB2x
    bsearchPosSortedGPU(sim->d_t1grid, sim->d_phi2, sim->d_posB2xOld, (*pool)[1]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("bsearchPosSortedGPU(phi2)");

    // cudaDeviceSynchronize(); //removed  // Ensure all kernels are complete before returning

    if (!config.log_response_interp) {
        // Linear-domain LN3
        indexVecLN3GPU(sim->d_weightsA1y, sim->d_indsA1y, sim->d_QKv, sim->d_QRv, len, sim->d_QKA1int, sim->d_QRA1int, (*pool)[2]);
        if (config.debug) DMFE_CUDA_POSTLAUNCH("indexVecLN3GPU A1");
        indexVecLN3GPU(sim->d_weightsA2y, sim->d_indsA2y, sim->d_QKv, sim->d_QRv, len, sim->d_QKA2int, sim->d_QRA2int, (*pool)[3]);
        if (config.debug) DMFE_CUDA_POSTLAUNCH("indexVecLN3GPU A2");
    } else {
        // Log-domain LN3: precompute separate log slices into temp11 (A1 stream) and temp12 (A2 stream)
        prepareLN3LogSliceGPU_into(len, sim->d_QRv, sim->temp11, (*pool)[2]);
        if (config.debug) DMFE_CUDA_POSTLAUNCH("prepareLN3LogSliceGPU_into A1");
        prepareLN3LogSliceGPU_into(len, sim->d_QRv, sim->temp12, (*pool)[3]);
        if (config.debug) DMFE_CUDA_POSTLAUNCH("prepareLN3LogSliceGPU_into A2");
        indexVecLN3GPU_log_cached(sim->d_weightsA1y, sim->d_indsA1y, sim->d_QKv, sim->d_QRv, sim->temp11, len, sim->d_QKA1int, sim->d_QRA1int, (*pool)[2]);
        if (config.debug) DMFE_CUDA_POSTLAUNCH("indexVecLN3GPU_log_cached A1");
        indexVecLN3GPU_log_cached(sim->d_weightsA2y, sim->d_indsA2y, sim->d_QKv, sim->d_QRv, sim->temp12, len, sim->d_QKA2int, sim->d_QRA2int, (*pool)[3]);
        if (config.debug) DMFE_CUDA_POSTLAUNCH("indexVecLN3GPU_log_cached A2");
    }

    // Interpolate QKB1int and QRB1int
    diffNfloor(sim->d_posB1xOld, sim->Stemp0, sim->temp0, (*pool)[0]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("diffNfloor B1");
    if (!config.log_response_interp) {
        indexVecNGPU(sim->temp0, sim->Stemp0, sim->d_delta_t_ratio, sim->d_QKB1int, sim->d_QRB1int, sim->d_QKv, sim->d_QRv, sim->d_dQKv, sim->d_dQRv, len, (*pool)[0]);
    } else {
        indexVecNGPU_log(sim->temp0, sim->Stemp0, sim->d_delta_t_ratio, sim->d_QKB1int, sim->d_QRB1int, sim->d_QKv, sim->d_QRv, sim->d_dQKv, sim->d_dQRv, len, (*pool)[0]);
    }
    if (config.debug) DMFE_CUDA_POSTLAUNCH("indexVecNGPU B1");

    // cudaDeviceSynchronize(); //removed  // Ensure all kernels are complete before returning

    // Interpolate QKB2int and QRB2int
    if (!config.log_response_interp) {
        indexMatAllGPU(sim->d_posB2xOld, sim->d_indsB2y, sim->d_weightsB2y, sim->d_delta_t_ratio, sim->d_QKB2int, sim->d_QRB2int, sim->d_QKv, sim->d_QRv, sim->d_dQKv, sim->d_dQRv, len, (*pool)[1]);
    } else {
        indexMatAllGPU_log(sim->d_posB2xOld, sim->d_indsB2y, sim->d_weightsB2y, sim->d_delta_t_ratio, sim->d_QKB2int, sim->d_QRB2int, sim->d_QKv, sim->d_QRv, sim->d_dQKv, sim->d_dQRv, len, (*pool)[1]);
    }
    if (config.debug) DMFE_CUDA_POSTLAUNCH("indexMatAllGPU B2");

    // Interpolate rInt
    indexVecR2GPU(sim->d_rvec, sim->d_drvec, sim->temp0, sim->Stemp0, sim->d_delta_t_ratio, sim->d_rInt, (*pool)[0]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("indexVecR2GPU");

    // cudaDeviceSynchronize(); //removed  // Ensure all kernels are complete before returning

    computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[2]>>>(
        sim->d_QKA1int.data().get(),
        sim->d_QRA1int.data().get(),
        sim->d_SigmaKA1int.data().get(),
        sim->d_SigmaRA1int.data().get(),
        len*len);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("computeSigmaKandRKernel A1");
    computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[3]>>>(
        sim->d_QKA2int.data().get(),
        sim->d_QRA2int.data().get(),
        sim->d_SigmaKA2int.data().get(),
        sim->d_SigmaRA2int.data().get(),
        len*len);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("computeSigmaKandRKernel A2");
    computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[0]>>>(
        sim->d_QKB1int.data().get(),
        sim->d_QRB1int.data().get(),
        sim->d_SigmaKB1int.data().get(),
        sim->d_SigmaRB1int.data().get(),
        len*len);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("computeSigmaKandRKernel B1");
    computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[1]>>>(
        sim->d_QKB2int.data().get(),
        sim->d_QRB2int.data().get(),
        sim->d_SigmaKB2int.data().get(),
        sim->d_SigmaRB2int.data().get(),
        len*len);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("computeSigmaKandRKernel B2");

    cudaDeviceSynchronize(); // Ensure all kernels are complete before returning
}
