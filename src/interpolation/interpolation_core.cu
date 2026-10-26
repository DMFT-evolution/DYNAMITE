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
#include "core/gpu_events.hpp"
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
    // [PERF] the four SigmaK/SigmaR launches that used their own <<<blocks,threads>>>
    // are now one batched launch; the block geometry lives in
    // computeSigmaKandRBatchedGPU().

    // [PERF] Remember whether the caller named a stream pool.  The rare callers
    // that do not (initialization, sparsification, the run controller) fall back
    // to getDefaultStreamPool(), a *different* set of streams from the one the
    // Runge-Kutta step uses; for them the closing device-wide barrier is kept, so
    // that their results are published to every stream exactly as before.
    const bool anonymousPool = (pool == nullptr);
    if (!pool) pool = &getDefaultStreamPool();

    // [PERF] r(t) is now written into rvec[idx] by a kernel on pool[3] instead of
    // travelling device->host->device (see rstepGPU_into in time_steps.cu).  The
    // blocking copies used to order the whole device against that write; the four
    // chains below read rvec (stream 0), QKv/QRv/dQKv/dQRv and delta_t_ratio, all
    // of which are produced upstream of that same event, so one wait per chain
    // reproduces the ordering the barrier gave for free.  Waiting on an event that
    // has never been recorded is a documented no-op, so the first call is safe.
    {
        const cudaEvent_t rstepOut = dmfe::EventPool::instance()[dmfe::EV_RSTEP_OUT];
        cudaStreamWaitEvent((*pool)[0], rstepOut, 0);
        cudaStreamWaitEvent((*pool)[1], rstepOut, 0);
        cudaStreamWaitEvent((*pool)[2], rstepOut, 0);
        cudaStreamWaitEvent((*pool)[3], rstepOut, 0);
    }

    // Compute sim->device->posB1x and sim->device->posB2x
    // [PERF] the theta search (L elements, 8 blocks -- almost pure launch latency)
    // and the phi2 search (L*L elements) are one launch over the concatenated index
    // space; see bsearch_interp_pair_kernel in search_utils.cu.  It runs on pool[0],
    // so the pool[1] chain (indexMatAll, which reads posB2xOld) is given the
    // dependency explicitly.
    bsearchPosSortedPairGPU(sim->device->t1grid,
                            sim->device->theta, sim->device->posB1xOld,
                            sim->device->phi2,  sim->device->posB2xOld,
                            (*pool)[0]);
    if (config.debug) DMFE_CUDA_POSTLAUNCH("bsearchPosSortedPairGPU(theta,phi2)");
    cudaEventRecord(dmfe::EventPool::instance()[dmfe::EV_BSEARCH], (*pool)[0]);
    cudaStreamWaitEvent((*pool)[1], dmfe::EventPool::instance()[dmfe::EV_BSEARCH], 0);

    // cudaDeviceSynchronize(); //removed  // Ensure all kernels are complete before returning

    if (!config.log_response_interp) {
        // Linear-domain LN3
        // [PERF] the A1 and A2 slices are interpolated by one launch with a
        // (blocks x 2) grid instead of two launches on two streams; see
        // indexVecLN3Kernel_batched in index_vec.cu.  When the shapes differ or an
        // unrolled kernel is preferable the helper falls back to the two original
        // launches on pool[2] and pool[3].
        indexVecLN3GPU_pair(
            sim->device->weightsA1y, sim->device->indsA1y, sim->device->QKA1int, sim->device->QRA1int, (*pool)[2],
            sim->device->weightsA2y, sim->device->indsA2y, sim->device->QKA2int, sim->device->QRA2int, (*pool)[3],
            sim->device->QKv, sim->device->QRv, len, (*pool)[2]);
        if (config.debug) DMFE_CUDA_POSTLAUNCH("indexVecLN3GPU_pair A1/A2");
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

    // [PERF] The join that used to live at the end of this function is done here
    // instead, because the four SigmaK/SigmaR evaluations below are now a single
    // launch on pool[0] and must therefore see the A1/A2/B2 slices produced on
    // pool[2], pool[3] and pool[1].  This function issues four independent chains
    // -- pool[0]: posB1xOld -> diffNfloor/temp0 -> indexVecN (B1) -> indexVecR2
    // (rInt); pool[1]: posB2xOld -> indexMatAll (B2); pool[2] and pool[3]: the A1
    // and A2 slices -- so funnelling 1, 2 and 3 into 0 reproduces exactly the
    // ordering the closing cudaDeviceSynchronize() provided, without stalling the
    // host.  Nothing about the kernels themselves changes, so nor do the results.
    {
        dmfe::EventPool& ev = dmfe::EventPool::instance();
        cudaEventRecord(ev[dmfe::EV_INTERP_S1], (*pool)[1]);
        cudaEventRecord(ev[dmfe::EV_INTERP_S2], (*pool)[2]);
        cudaEventRecord(ev[dmfe::EV_INTERP_S3], (*pool)[3]);
        cudaStreamWaitEvent((*pool)[0], ev[dmfe::EV_INTERP_S1], 0);
        cudaStreamWaitEvent((*pool)[0], ev[dmfe::EV_INTERP_S2], 0);
        cudaStreamWaitEvent((*pool)[0], ev[dmfe::EV_INTERP_S3], 0);
    }

    // [PERF] four element-wise launches (A1, A2, B1, B2) of L*L threads each,
    // previously one per stream, are now one launch with a (blocks x 4) grid.
    // Every output element is produced by the same expression from the same two
    // inputs; see the note on SigmaBatchDesc in math_sigma.cuh.
    {
        SigmaBatchDesc sb;
        auto P = [](const dmfe::device_vector<double>& v) { return thrust::raw_pointer_cast(v.data()); };
        auto Q = [](dmfe::device_vector<double>& v) { return thrust::raw_pointer_cast(v.data()); };
        sb.add(P(sim->device->QKA1int), P(sim->device->QRA1int), Q(sim->device->SigmaKA1int), Q(sim->device->SigmaRA1int));
        sb.add(P(sim->device->QKA2int), P(sim->device->QRA2int), Q(sim->device->SigmaKA2int), Q(sim->device->SigmaRA2int));
        sb.add(P(sim->device->QKB1int), P(sim->device->QRB1int), Q(sim->device->SigmaKB1int), Q(sim->device->SigmaRB1int));
        sb.add(P(sim->device->QKB2int), P(sim->device->QRB2int), Q(sim->device->SigmaKB2int), Q(sim->device->SigmaRB2int));
        computeSigmaKandRBatchedGPU(sb, len * len, (*pool)[0]);
    }
    if (config.debug) DMFE_CUDA_POSTLAUNCH("computeSigmaKandRBatchedGPU A1/A2/B1/B2");

    // [PERF] For the rare callers that did not name a stream pool (initialization,
    // sparsification, the run controller) the closing device-wide barrier is kept,
    // so their results are published to every stream exactly as before.  For the
    // Runge-Kutta path the join above is enough: the single consumer, QKRstepGPU,
    // funnels everything through pool[0] (thrust::for_each -> ConvBatchedGPU -> one
    // event into pool[9..11]).
    if (anonymousPool || config.debug) {
        cudaDeviceSynchronize(); // Ensure all kernels are complete before returning
    }
}
