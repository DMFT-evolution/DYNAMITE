#include "interpolation_core.hpp"
#include "config.hpp"
#include "simulation_data.hpp"
#include "search_utils.hpp"
#include "index_vec.hpp"
#include "index_mat.hpp"
#include "math_sigma.hpp"
#include "vector_utils.hpp"
#include "stream_pool.hpp"
#include "io_utils.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <thrust/device_vector.h>

using namespace std;

// External declarations for global variables
extern SimulationConfig config;
extern SimulationData* sim;

void interpolate(const vector<double>& posB1xIn, const vector<double>& posB2xIn,
    const bool same)
{
    // Compute posB1x
    vector<double> posB1x = !posB1xIn.empty() ?
        (same ? posB1xIn : isearchPosSortedInit(sim->h_t1grid, sim->h_theta, posB1xIn)) :
        bsearchPosSorted(sim->h_t1grid, sim->h_theta * sim->h_t1grid.back());

    // Compute posB2x
    vector<double> posB2x = !posB2xIn.empty() ?
        (same ? posB2xIn : isearchPosSortedInit(sim->h_t1grid, sim->h_theta, posB2xIn)) :
    bsearchPosSorted(sim->h_t1grid, sim->h_phi2 * sim->h_t1grid.back());

    // Update old positions
    sim->h_posB1xOld = posB1x;
    sim->h_posB2xOld = posB2x;

    // Interpolate QKA1int and QRA1int
    if (sim->h_t1grid.back() > 0) {
        indexVecLN3(sim->h_weightsA1y, sim->h_indsA1y, sim->h_QKA1int, sim->h_QRA1int, config.len);
    }
    else {
        sim->h_QKA1int.assign(config.len * config.len, sim->h_QKv[0]);
        sim->h_QRA1int.assign(config.len * config.len, sim->h_QRv[0]);
    }
    SigmaK(sim->h_QKA1int, sim->h_SigmaKA1int);
    SigmaR(sim->h_QKA1int, sim->h_QRA1int, sim->h_SigmaRA1int);

    // Interpolate QKA2int and QRA2int
    if (sim->h_t1grid.back() > 0) {
        indexVecLN3(sim->h_weightsA2y, sim->h_indsA2y, sim->h_QKA2int, sim->h_QRA2int, config.len);
    }
    else {
        sim->h_QKA2int.assign(config.len * config.len, sim->h_QKv[0]);
        sim->h_QRA2int.assign(config.len * config.len, sim->h_QRv[0]);
    }
    SigmaR(sim->h_QKA2int, sim->h_QRA2int, sim->h_SigmaRA2int);

    // Interpolate QKB1int and QRB1int
    // Compute `floor` vector
    double maxPosB1x = posB1x[0];
    for (size_t i = 1; i < posB1x.size(); ++i) {
        if (posB1x[i] > maxPosB1x) {
            maxPosB1x = posB1x[i];
        }
    }
    size_t maxCeil = static_cast<size_t>(ceil(maxPosB1x)) - 1;
    if (maxCeil < 1) {
        maxCeil = 1;
    }

    // Compute FLOOR vector
    vector<size_t> Floor(posB1x.size());
    for (size_t i = 0; i < posB1x.size(); ++i) {
        size_t flooredValue = static_cast<size_t>(floor(posB1x[i]));
        if (flooredValue < 1) {
            flooredValue = 1;
        }
        else if (flooredValue > maxCeil) {
            flooredValue = maxCeil;
        }
        Floor[i] = flooredValue;
    }

    // Compute `diff` vector
    vector<double> diff(posB1x.size());
    // Subtract(vector<double>(Floor.begin(), Floor.end()), posB1x, diff);
    diff = vector<double>(Floor.begin(), Floor.end()) - posB1x;

    if (sim->h_t1grid.back() > 0) {
        indexVecN(config.len, diff, Floor, sim->h_delta_t_ratio, sim->h_QKB1int, sim->h_QRB1int, config.len);
    }
    else {
        sim->h_QKB1int.assign(config.len * config.len, sim->h_QKv[0]);
        sim->h_QRB1int.assign(config.len * config.len, sim->h_QRv[0]);
    }
    SigmaK(sim->h_QKB1int, sim->h_SigmaKB1int);
    SigmaR(sim->h_QKB1int, sim->h_QRB1int, sim->h_SigmaRB1int);

    // Interpolate QKB2int and QRB2int
    if (sim->h_t1grid.back() > 0) {
        indexMatAll(sim->h_posB2xOld, sim->h_indsB2y, sim->h_weightsB2y, sim->h_delta_t_ratio, sim->h_QKB2int, sim->h_QRB2int);
    }
    else {
        sim->h_QKB2int.assign(config.len * config.len, sim->h_QKv[0]);
        sim->h_QRB2int.assign(config.len * config.len, sim->h_QRv[0]);
    }
    SigmaK(sim->h_QKB2int, sim->h_SigmaKB2int);
    SigmaR(sim->h_QKB2int, sim->h_QRB2int, sim->h_SigmaRB2int);

    // Interpolate rInt
    if (sim->h_t1grid.back() > 0) {
        indexVecR2(sim->h_rvec, sim->h_drvec, diff, Floor, sim->h_delta_t_ratio, sim->h_rInt);
    }
    else {
    sim->h_rInt.assign(config.len, sim->h_rvec[0]);
    }
}

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

    // Compute sim->d_posB2x
    bsearchPosSortedGPU(sim->d_t1grid, sim->d_phi2, sim->d_posB2xOld, (*pool)[1]);

    // cudaDeviceSynchronize(); //removed  // Ensure all kernels are complete before returning

    // Interpolate QKA1int and QRA1int using centralized LN3 GPU path
    indexVecLN3GPU(sim->d_weightsA1y, sim->d_indsA1y, sim->d_QKv, sim->d_QRv, len, sim->d_QKA1int, sim->d_QRA1int, (*pool)[2]);

    // Interpolate QKA2int and QRA2int using centralized LN3 GPU path
    indexVecLN3GPU(sim->d_weightsA2y, sim->d_indsA2y, sim->d_QKv, sim->d_QRv, len, sim->d_QKA2int, sim->d_QRA2int, (*pool)[3]);

    // Interpolate QKB1int and QRB1int
    diffNfloor(sim->d_posB1xOld, sim->Stemp0, sim->temp0, (*pool)[0]);
    indexVecNGPU(sim->temp0, sim->Stemp0, sim->d_delta_t_ratio, sim->d_QKB1int, sim->d_QRB1int, sim->d_QKv, sim->d_QRv, sim->d_dQKv, sim->d_dQRv, len, (*pool)[0]);

    cudaDeviceSynchronize();  // Ensure all kernels are complete before returning

    // Interpolate QKB2int and QRB2int
    indexMatAllGPU(sim->d_posB2xOld, sim->d_indsB2y, sim->d_weightsB2y, sim->d_delta_t_ratio, sim->d_QKB2int, sim->d_QRB2int, sim->d_QKv, sim->d_QRv, sim->d_dQKv, sim->d_dQRv, len, (*pool)[1]);

    // Interpolate rInt
    indexVecR2GPU(sim->d_rvec, sim->d_drvec, sim->temp0, sim->Stemp0, sim->d_delta_t_ratio, sim->d_rInt, (*pool)[0]);

    // cudaDeviceSynchronize(); //removed  // Ensure all kernels are complete before returning

    computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[2]>>>(
        sim->d_QKA1int.data().get(),
        sim->d_QRA1int.data().get(),
        sim->d_SigmaKA1int.data().get(),
        sim->d_SigmaRA1int.data().get(),
        len*len);
    computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[3]>>>(
        sim->d_QKA2int.data().get(),
        sim->d_QRA2int.data().get(),
        sim->d_SigmaKA2int.data().get(),
        sim->d_SigmaRA2int.data().get(),
        len*len);
    computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[0]>>>(
        sim->d_QKB1int.data().get(),
        sim->d_QRB1int.data().get(),
        sim->d_SigmaKB1int.data().get(),
        sim->d_SigmaRB1int.data().get(),
        len*len);
    computeSigmaKandRKernel<<<blocks, threads, 0, (*pool)[1]>>>(
        sim->d_QKB2int.data().get(),
        sim->d_QRB2int.data().get(),
        sim->d_SigmaKB2int.data().get(),
        sim->d_SigmaRB2int.data().get(),
        len*len);

    // cudaDeviceSynchronize(); //removed  // Ensure all kernels are complete before returning
}
