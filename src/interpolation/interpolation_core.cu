#include "interpolation/interpolation_core.hpp"
#include "core/config.hpp"
#include "simulation/simulation_data.hpp"
#include "simulation/device_simulation_data.hpp"
#include "search/search_utils.cuh"
#include "interpolation/index_vec.cuh"
#include "interpolation/index_mat.cuh"
#include "math/math_sigma.cuh"
#include "core/vector_utils.cuh"
#include "core/stream_pool.hpp"
#include "io/io_utils.hpp"
#include "core/device_utils.cuh"
#include <vector>
#include <cmath>
#include <algorithm>

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
    const double* posB1x,
    size_t* Floor,
    double* diff,
    size_t len,
    cudaStream_t stream)
{
    constexpr int threads = 64;
    const int blocks = static_cast<int>((len + threads - 1) / threads);

    diffNfloorKernel<<<blocks, threads, 0, stream>>>(
        posB1x,
        Floor,
        diff,
        len
    );
}

void interpolateGPU(
    const double* posB1xIn,
    const double* posB2xIn,
    const bool same,
    StreamPool* pool) {

    size_t len = sim->host->theta.size();
    int threads = 64;
    int blocks = (len*len + threads - 1) / threads;

    if (!pool) pool = &getDefaultStreamPool();

    // Compute sim->device->posB1x
    bsearchPosSortedGPU(sim->device->t1grid, sim->device->theta, sim->device->posB1xOld, (*pool)[0]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("bsearchPosSortedGPU(theta)");

    // Compute sim->device->posB2x
    bsearchPosSortedGPU(sim->device->t1grid, sim->device->phi2, sim->device->posB2xOld, (*pool)[1]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("bsearchPosSortedGPU(phi2)");

    // cudaDeviceSynchronize(); //removed  // Ensure all kernels are complete before returning

    if (!config.log_response_interp) {
        // Linear-domain LN3
        indexVecLN3GPU(sim->device->weightsA1y, sim->device->indsA1y, sim->device->QKv, sim->device->QRv, len, sim->device->QKA1int, sim->device->QRA1int, (*pool)[2]);
        if (config.debug) DMFE_CUDA_POSTLAUNCH("indexVecLN3GPU A1");
        indexVecLN3GPU(sim->device->weightsA2y, sim->device->indsA2y, sim->device->QKv, sim->device->QRv, len, sim->device->QKA2int, sim->device->QRA2int, (*pool)[3]);
        if (config.debug) DMFE_CUDA_POSTLAUNCH("indexVecLN3GPU A2");
    } else {
        // Log-domain LN3: precompute separate log slices into temp11 (A1 stream) and temp12 (A2 stream)
        prepareLN3LogSliceGPU_into(len, sim->device->QRv, sim->device->temp11, (*pool)[2]);
        if (config.debug) DMFE_CUDA_POSTLAUNCH("prepareLN3LogSliceGPU_into A1");
        prepareLN3LogSliceGPU_into(len, sim->device->QRv, sim->device->temp12, (*pool)[3]);
        if (config.debug) DMFE_CUDA_POSTLAUNCH("prepareLN3LogSliceGPU_into A2");
        indexVecLN3GPU_log_cached(sim->device->weightsA1y, sim->device->indsA1y, sim->device->QKv, sim->device->QRv, sim->device->temp11, len, sim->device->QKA1int, sim->device->QRA1int, (*pool)[2]);
        if (config.debug) DMFE_CUDA_POSTLAUNCH("indexVecLN3GPU_log_cached A1");
        indexVecLN3GPU_log_cached(sim->device->weightsA2y, sim->device->indsA2y, sim->device->QKv, sim->device->QRv, sim->device->temp12, len, sim->device->QKA2int, sim->device->QRA2int, (*pool)[3]);
        if (config.debug) DMFE_CUDA_POSTLAUNCH("indexVecLN3GPU_log_cached A2");
    }

    // Interpolate QKB1int and QRB1int
    diffNfloor(
    thrust::raw_pointer_cast(sim->device->posB1xOld.data()),
    thrust::raw_pointer_cast(sim->device->Stemp0.data()),
    thrust::raw_pointer_cast(sim->device->temp0.data()),
    sim->device->posB1xOld.size(),
    (*pool)[0]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("diffNfloor B1");
    if (!config.log_response_interp) {
        indexVecNGPU(sim->device->temp0, sim->device->Stemp0, sim->device->delta_t_ratio, sim->device->QKB1int, sim->device->QRB1int, sim->device->QKv, sim->device->QRv, sim->device->dQKv, sim->device->dQRv, len, (*pool)[0]);
    } else {
        indexVecNGPU_log(sim->device->temp0, sim->device->Stemp0, sim->device->delta_t_ratio, sim->device->QKB1int, sim->device->QRB1int, sim->device->QKv, sim->device->QRv, sim->device->dQKv, sim->device->dQRv, len, (*pool)[0]);
    }
    if (config.debug) DMFE_CUDA_POSTLAUNCH("indexVecNGPU B1");

    // cudaDeviceSynchronize(); //removed  // Ensure all kernels are complete before returning

    // Interpolate QKB2int and QRB2int
    if (!config.log_response_interp) {
        indexMatAllGPU(sim->device->posB2xOld, sim->device->indsB2y, sim->device->weightsB2y, sim->device->delta_t_ratio, sim->device->QKB2int, sim->device->QRB2int, sim->device->QKv, sim->device->QRv, sim->device->dQKv, sim->device->dQRv, len, (*pool)[1]);
    } else {
        indexMatAllGPU_log(sim->device->posB2xOld, sim->device->indsB2y, sim->device->weightsB2y, sim->device->delta_t_ratio, sim->device->QKB2int, sim->device->QRB2int, sim->device->QKv, sim->device->QRv, sim->device->dQKv, sim->device->dQRv, len, (*pool)[1]);
    }
    if (config.debug) DMFE_CUDA_POSTLAUNCH("indexMatAllGPU B2");

    // Interpolate rInt
    indexVecR2GPU(sim->device->rvec, sim->device->drvec, sim->device->temp0, sim->device->Stemp0, sim->device->delta_t_ratio, sim->device->rInt, (*pool)[0]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("indexVecR2GPU");

    // cudaDeviceSynchronize(); //removed  // Ensure all kernels are complete before returning

    computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[2]>>>(
        thrust::raw_pointer_cast(sim->device->QKA1int.data()),
        thrust::raw_pointer_cast(sim->device->QRA1int.data()),
        thrust::raw_pointer_cast(sim->device->SigmaKA1int.data()),
        thrust::raw_pointer_cast(sim->device->SigmaRA1int.data()),
        len*len);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("computeSigmaKandRKernel A1");
    computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[3]>>>(
        thrust::raw_pointer_cast(sim->device->QKA2int.data()),
        thrust::raw_pointer_cast(sim->device->QRA2int.data()),
        thrust::raw_pointer_cast(sim->device->SigmaKA2int.data()),
        thrust::raw_pointer_cast(sim->device->SigmaRA2int.data()),
        len*len);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("computeSigmaKandRKernel A2");
    computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[0]>>>(
        thrust::raw_pointer_cast(sim->device->QKB1int.data()),
        thrust::raw_pointer_cast(sim->device->QRB1int.data()),
        thrust::raw_pointer_cast(sim->device->SigmaKB1int.data()),
        thrust::raw_pointer_cast(sim->device->SigmaRB1int.data()),
        len*len);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("computeSigmaKandRKernel B1");
    computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[1]>>>(
        thrust::raw_pointer_cast(sim->device->QKB2int.data()),
        thrust::raw_pointer_cast(sim->device->QRB2int.data()),
        thrust::raw_pointer_cast(sim->device->SigmaKB2int.data()),
        thrust::raw_pointer_cast(sim->device->SigmaRB2int.data()),
        len*len);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("computeSigmaKandRKernel B2");

    cudaDeviceSynchronize(); // Ensure all kernels are complete before returning
}
