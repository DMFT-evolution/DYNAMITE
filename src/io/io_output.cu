#include "io/io_utils.hpp"
#include "simulation/simulation_data.hpp"
#include "simulation/device_simulation_data.hpp"
#include "core/gpu_memory_utils.hpp"
#include "core/backend_dispatch.hpp"
#include "math/math_ops.hpp"
#include "math/math_sigma.cuh"
#include "core/config.hpp"
#include "core/globals.hpp"

#include "convolution/convolution.cuh"
#include "core/device_utils.cuh"
#include "EOMs/time_steps.hpp"
#include "EOMs/rk_data.hpp"
#include "EOMs/device_rk_data.hpp"
#include "version/version_info.hpp"
#include <fstream>
#include <iostream>
#include "core/console.hpp"
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <thrust/copy.h>
#include <unistd.h>
#include <limits.h>
#include <errno.h>
#include <cstring>
#include <chrono>
#include <ctime>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>

#if defined(H5_RUNTIME_OPTIONAL)
#include "io/h5_runtime.hpp"
#elif defined(USE_HDF5)
#include "H5Cpp.h"
#endif

#include "version/version_compat.hpp"

using namespace std;

// External global variables
extern SimulationConfig config;
extern SimulationData* sim;
extern RKData* rk;
extern size_t peak_memory_mb;
extern size_t peak_gpu_memory_mb;
extern std::chrono::high_resolution_clock::time_point program_start_time;

// Global variables for async save synchronization
extern std::mutex saveMutex;
extern bool saveInProgress;
extern std::condition_variable saveCondition;

// Forward declarations
void saveCompressedDataAsync(const std::string& dirPath, const SimulationDataSnapshot& snapshot);

// GPU kernel for computing energy at all time points efficiently
__global__ void computeEnergyHistoryKernel(
    const double* __restrict__ QKv,
    const double* __restrict__ QRv,
    const double* __restrict__ integ,
    const double* __restrict__ theta,
    const double* __restrict__ t1grid,
    double* __restrict__ energy_history,
    double T0,
    size_t len,
    size_t t1len)
{
    extern __shared__ double sdata[];
    double* integ_shared = sdata;
    double* reduction_shared = &sdata[len];
    
    int t_idx = blockIdx.x;  // Each block handles one time point
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    
    if (t_idx >= t1len) return;
    
    // Load integ into shared memory once per block
    for (int i = tid; i < len; i += nthreads) {
        integ_shared[i] = integ[i];
    }
    __syncthreads();
    
    // Compute sigmaK and convolution in one pass
    double sum = 0.0;
    double t = t1grid[t_idx];
    size_t base_idx = t_idx * len;
    
    // [CPC-REV] BUG FIX. This kernel (used only for the GPU energy history written
    // to energy.txt) still evaluated the beta_0 contribution of Eq. (14) as
    // f'(C(t,0))/T0 and, in addition, multiplied it by t together with the memory
    // integral.  The correct expression is
    //     E(t) = -[ t * sum_i nu_i f'(C(t,theta_i t)) R(t,theta_i t) + f(C(t,0))/T0 ]
    // i.e. f and not f', and no factor t on the beta_0 term.  The CPU path
    // (energyCPU(), src/EOMs/time_steps.cpp) and the GPU scalar energyGPU() were
    // corrected in commit de696f84; this kernel was missed, so CPU and GPU energy
    // histories disagreed by [t f'(C(t,0)) - f(C(t,0))]/T0 whenever T0 was finite.
    // For the T0 = inf runs of the manuscript the term is ~1e-50 and both agree.
    for (int i = tid; i < len; i += nthreads) {
        double qk = QKv[base_idx + i];
        double qr = QRv[base_idx + i];
        double sigmaK = DflambdaGPU(qk);

        // Match the ConvAGPU scaling approach
        sum += sigmaK * qr * integ_shared[i];
    }
    
    // Block-level reduction
    reduction_shared[tid] = sum;
    __syncthreads();
    
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction_shared[tid] += reduction_shared[tid + stride];
        }
        __syncthreads();
    }
    
    // Write final energy (negative because it's -energy)
    if (tid == 0) {
        // [CPC-REV] see the note above: beta_0 term added outside the t prefactor
        // and evaluated with f, not f'.
        energy_history[t_idx] = -(t * reduction_shared[0] + flambdaGPU(QKv[base_idx]) / T0);
    }
}

// Function to create a snapshot of simulation data for background saving
SimulationDataSnapshot createDataSnapshot()
{
    SimulationDataSnapshot snapshot;
    
    // Copy vectors - directly from device for GPU runs to avoid unnecessary host copies
    if (config.gpu) {
        // Direct copy from device vectors to avoid double copying
        auto copyDeviceToSnapshot = [](std::vector<double>& snapshot_vec, const dmfe::device_vector<double>& device_vec) {
            snapshot_vec.resize(device_vec.size());
            thrust::copy(device_vec.begin(), device_vec.end(), snapshot_vec.begin());
        };
        
        copyDeviceToSnapshot(snapshot.QKv, sim->device->QKv);
        copyDeviceToSnapshot(snapshot.QRv, sim->device->QRv);
        copyDeviceToSnapshot(snapshot.dQKv, sim->device->dQKv);
        copyDeviceToSnapshot(snapshot.dQRv, sim->device->dQRv);
        copyDeviceToSnapshot(snapshot.t1grid, sim->device->t1grid);
        copyDeviceToSnapshot(snapshot.rvec, sim->device->rvec);
        copyDeviceToSnapshot(snapshot.drvec, sim->device->drvec);
        
        // Copy compressed data from device
        copyDeviceToSnapshot(snapshot.QKB1int, sim->device->QKB1int);
        copyDeviceToSnapshot(snapshot.QRB1int, sim->device->QRB1int);
        copyDeviceToSnapshot(snapshot.theta, sim->device->theta);
        snapshot.t_current = snapshot.t1grid.back();
        snapshot.method = rk->device->init; // Store the RK method used on GPU
    } else {
        // For CPU runs, copy from host vectors as before
        snapshot.QKv = sim->host->QKv;
        snapshot.QRv = sim->host->QRv;
        snapshot.dQKv = sim->host->dQKv;
        snapshot.dQRv = sim->host->dQRv;
        snapshot.t1grid = sim->host->t1grid;
        snapshot.rvec = sim->host->rvec;
        snapshot.drvec = sim->host->drvec;
        
        // Copy compressed data
        snapshot.QKB1int = sim->host->QKB1int;
        snapshot.QRB1int = sim->host->QRB1int;
        snapshot.theta = sim->host->theta;
        snapshot.t_current = sim->host->t1grid.back();
        snapshot.method = rk->host->init;  // Store the RK method used on CPU
    }
    
    // Calculate energy
    if (config.gpu) {
        snapshot.energy = energy();
    } else {
        std::vector<double> temp(config.len, 0.0);
        std::vector<double> lastQKv = getLastLenEntries(sim->host->QKv, config.len);
        SigmaK(lastQKv, temp);
        snapshot.energy = -(ConvA(temp, getLastLenEntries(sim->host->QRv, config.len), sim->host->t1grid.back())[0] + flambda(lastQKv[0])/config.T0);
    }
    
    // Copy metadata

    snapshot.current_len = config.len;
    snapshot.current_loop = config.loop;
    snapshot.config_snapshot = config;
    
    // Capture version information
    extern VersionInfo g_version_info;
    snapshot.code_version = g_version_info.code_version;
    snapshot.git_hash = g_version_info.git_hash;
    snapshot.git_branch = g_version_info.git_branch;
    snapshot.git_tag = g_version_info.git_tag;
    snapshot.git_dirty = g_version_info.git_dirty;
    snapshot.build_date = g_version_info.build_date;
    snapshot.build_time = g_version_info.build_time;
    snapshot.compiler_version = g_version_info.compiler_version;
    snapshot.cuda_version = g_version_info.cuda_version;
    
    // Capture memory statistics
    extern size_t peak_memory_mb;
    extern size_t peak_gpu_memory_mb;
    extern std::chrono::high_resolution_clock::time_point program_start_time;
    snapshot.peak_memory_mb_snapshot = peak_memory_mb;
    snapshot.peak_gpu_memory_mb_snapshot = peak_gpu_memory_mb;
    snapshot.program_start_time_snapshot = program_start_time;

    // Capture debug runtime telemetry (host-only data)
    snapshot.debug_step_times = sim->host->debug_step_times;
    snapshot.debug_step_runtimes = sim->host->debug_step_runtimes;
    snapshot.debug_step_memory = sim->host->debug_step_memory;
    
    return snapshot;
}

// Energy E(t_n) for all stored times on the GPU; used by saveHistory()
// (src/io/io_save_history.cpp).
void energyHistoryGPU(SimulationData& simulation, size_t len_param, double T0_param,
                      std::vector<double>& energy_history)
{
    const size_t t1len = simulation.device->t1grid.size();
    dmfe::device_vector<double> d_energy_history(t1len);

    int threads = 64;
    size_t shmem = len_param * sizeof(double) + threads * sizeof(double);

    computeEnergyHistoryKernel<<<t1len, threads, shmem>>>(
        thrust::raw_pointer_cast(simulation.device->QKv.data()),
        thrust::raw_pointer_cast(simulation.device->QRv.data()),
        thrust::raw_pointer_cast(simulation.device->integ.data()),
        thrust::raw_pointer_cast(simulation.device->theta.data()),
        thrust::raw_pointer_cast(simulation.device->t1grid.data()),
        thrust::raw_pointer_cast(d_energy_history.data()),
        T0_param, len_param, t1len
    );

    energy_history.resize(t1len);
    thrust::copy(d_energy_history.begin(), d_energy_history.end(), energy_history.begin());
}


