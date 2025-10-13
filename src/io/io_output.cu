#include "io/io_utils.hpp"
#include "simulation/simulation_data.hpp"
#include "core/gpu_memory_utils.hpp"
#include "math/math_ops.hpp"
#include "math/math_sigma.hpp"
#include "core/config.hpp"
#include "core/globals.hpp"
#include "convolution/convolution.hpp"
#include "core/device_utils.cuh"
#include "EOMs/time_steps.hpp"
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
#include <thrust/device_vector.h>
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
extern size_t peak_memory_kb;
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
    
    for (int i = tid; i < len; i += nthreads) {
        double qk = QKv[base_idx + i];
        double qr = QRv[base_idx + i];
        double sigmaK = DflambdaGPU(qk);
        
        // Match the ConvAGPU scaling approach
        sum += sigmaK * qr * integ_shared[i];
        
        // Add direct term for i=0
        if (i == 0) {
            sum += sigmaK / T0;
        }
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
        energy_history[t_idx] = -t * reduction_shared[0];
    }
}

// Function to create a snapshot of simulation data for background saving
SimulationDataSnapshot createDataSnapshot()
{
    SimulationDataSnapshot snapshot;
    
    // Copy vectors - directly from device for GPU runs to avoid unnecessary host copies
    if (config.gpu) {
        // Direct copy from device vectors to avoid double copying
        auto copyDeviceToSnapshot = [](std::vector<double>& snapshot_vec, const thrust::device_vector<double>& device_vec) {
            snapshot_vec.resize(device_vec.size());
            thrust::copy(device_vec.begin(), device_vec.end(), snapshot_vec.begin());
        };
        
        copyDeviceToSnapshot(snapshot.QKv, sim->d_QKv);
        copyDeviceToSnapshot(snapshot.QRv, sim->d_QRv);
        copyDeviceToSnapshot(snapshot.dQKv, sim->d_dQKv);
        copyDeviceToSnapshot(snapshot.dQRv, sim->d_dQRv);
        copyDeviceToSnapshot(snapshot.t1grid, sim->d_t1grid);
        copyDeviceToSnapshot(snapshot.rvec, sim->d_rvec);
        copyDeviceToSnapshot(snapshot.drvec, sim->d_drvec);
        
        // Copy compressed data from device
        copyDeviceToSnapshot(snapshot.QKB1int, sim->d_QKB1int);
        copyDeviceToSnapshot(snapshot.QRB1int, sim->d_QRB1int);
        copyDeviceToSnapshot(snapshot.theta, sim->d_theta);
        snapshot.t_current = snapshot.t1grid.back();
    } else {
        // For CPU runs, copy from host vectors as before
        snapshot.QKv = sim->h_QKv;
        snapshot.QRv = sim->h_QRv;
        snapshot.dQKv = sim->h_dQKv;
        snapshot.dQRv = sim->h_dQRv;
        snapshot.t1grid = sim->h_t1grid;
        snapshot.rvec = sim->h_rvec;
        snapshot.drvec = sim->h_drvec;
        
        // Copy compressed data
        snapshot.QKB1int = sim->h_QKB1int;
        snapshot.QRB1int = sim->h_QRB1int;
        snapshot.theta = sim->h_theta;
        snapshot.t_current = sim->h_t1grid.back();
    }
    
    // Calculate energy
    if (config.gpu) {
        snapshot.energy = energyGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, config.T0);
    } else {
        std::vector<double> temp(config.len, 0.0);
        std::vector<double> lastQKv = getLastLenEntries(sim->h_QKv, config.len);
        SigmaK(lastQKv, temp);
        snapshot.energy = -(ConvA(temp, getLastLenEntries(sim->h_QRv, config.len), sim->h_t1grid.back())[0] + Dflambda(lastQKv[0])/config.T0);
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
    extern size_t peak_memory_kb;
    extern size_t peak_gpu_memory_mb;
    extern std::chrono::high_resolution_clock::time_point program_start_time;
    snapshot.peak_memory_kb_snapshot = peak_memory_kb;
    snapshot.peak_gpu_memory_mb_snapshot = peak_gpu_memory_mb;
    snapshot.program_start_time_snapshot = program_start_time;
    
    return snapshot;
}

void saveHistory(const std::string& filename, double delta, double delta_t, 
                 SimulationData& simulation, size_t len_param, double T0_param, bool gpu_param)
{
    // Ensure data is on CPU for saving
    if (gpu_param) {
        copyVectorsToCPU(simulation);
    }
    
    size_t t1len = simulation.h_t1grid.size();
    std::vector<double> energy_history(t1len);
    std::vector<double> qk0_history(t1len);
    
    if (gpu_param) {
        // Use GPU to compute energy history efficiently
        thrust::device_vector<double> d_energy_history(t1len);

        int threads = 64;
        size_t shmem = len_param * sizeof(double) + threads * sizeof(double);

        computeEnergyHistoryKernel<<<t1len, threads, shmem>>>(
            thrust::raw_pointer_cast(simulation.d_QKv.data()),
            thrust::raw_pointer_cast(simulation.d_QRv.data()),
            thrust::raw_pointer_cast(simulation.d_integ.data()),
            thrust::raw_pointer_cast(simulation.d_theta.data()),
            thrust::raw_pointer_cast(simulation.d_t1grid.data()),
            thrust::raw_pointer_cast(d_energy_history.data()),
            T0_param, len_param, t1len
        );
        
        // Copy energy history back to CPU
        thrust::copy(d_energy_history.begin(), d_energy_history.end(), energy_history.begin());
        
        // Extract QK[0] values for each time step
        for (size_t i = 0; i < t1len; ++i) {
            qk0_history[i] = simulation.h_QKv[i * len_param];  // QKv[i * len] is QK[0] at time step i
        }
    } else {
        // CPU computation of energy history
        for (size_t i = 0; i < t1len; ++i) {
            std::vector<double> temp(len_param, 0.0);
            std::vector<double> QKv_i(simulation.h_QKv.begin() + i * len_param, simulation.h_QKv.begin() + (i + 1) * len_param);
            std::vector<double> QRv_i(simulation.h_QRv.begin() + i * len_param, simulation.h_QRv.begin() + (i + 1) * len_param);
            SigmaK(QKv_i, temp);
            energy_history[i] = -(ConvA(temp, QRv_i, simulation.h_t1grid[i])[0] + Dflambda(QKv_i[0]) / T0_param);
            qk0_history[i] = QKv_i[0];  // QK[0] at time step i
        }
    }
    
    // Get directory path from filename
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    
    // Save rvec history
    std::string rvecFilename = dirPath + "/rvec.txt";
    std::ofstream rvecFile(rvecFilename);
    if (rvecFile) {
        rvecFile << std::fixed << std::setprecision(16);
        rvecFile << "# Time\trvec\n";
        for (size_t i = 0; i < t1len; ++i) {
            rvecFile << simulation.h_t1grid[i] << "\t" << simulation.h_rvec[i] << "\n";
        }
        rvecFile.close();
    std::cout << dmfe::console::SAVE() << "Saved rvec history to " << rvecFilename << std::endl;
    } else {
    std::cerr << dmfe::console::ERR() << "Could not open file " << rvecFilename << std::endl;
    }
    
    // Save energy history
    std::string energyFilename = dirPath + "/energy.txt";
    std::ofstream energyFile(energyFilename);
    if (energyFile) {
        energyFile << std::fixed << std::setprecision(16);
        energyFile << "# Time\tEnergy\n";
        for (size_t i = 0; i < t1len; ++i) {
            energyFile << simulation.h_t1grid[i] << "\t" << energy_history[i] << "\n";
        }
        energyFile.close();
    std::cout << dmfe::console::SAVE() << "Saved energy history to " << energyFilename << std::endl;
    } else {
    std::cerr << dmfe::console::ERR() << "Could not open file " << energyFilename << std::endl;
    }
    
    // Save QK[0] history
    std::string qk0Filename = dirPath + "/qk0.txt";
    std::ofstream qk0File(qk0Filename);
    if (qk0File) {
        qk0File << std::fixed << std::setprecision(16);
        qk0File << "# Time\tQK[0]\n";
        for (size_t i = 0; i < t1len; ++i) {
            qk0File << simulation.h_t1grid[i] << "\t" << qk0_history[i] << "\n";
        }
        qk0File.close();
    std::cout << dmfe::console::SAVE() << "Saved QK[0] history to " << qk0Filename << std::endl;
    } else {
    std::cerr << dmfe::console::ERR() << "Could not open file " << qk0Filename << std::endl;
    }
    
    std::cout << dmfe::console::SAVE() << "Successfully saved complete history (" << t1len << " time points) to " << dirPath << std::endl;
}

// Async version of saveHistory that works with snapshot data (simplified, no GPU operations)
void saveHistoryAsync(const std::string& filename, double delta, double delta_t, const SimulationDataSnapshot& snapshot)
{
    // Get directory path from filename
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    
    size_t t1len = snapshot.t1grid.size();
    std::vector<double> energy_history(t1len);
    
    // CPU computation of energy history (same as synchronous version)
    for (size_t i = 0; i < t1len; ++i) {
        std::vector<double> temp(snapshot.current_len, 0.0);
        std::vector<double> QKv_i(snapshot.QKv.begin() + i * snapshot.current_len, snapshot.QKv.begin() + (i + 1) * snapshot.current_len);
        std::vector<double> QRv_i(snapshot.QRv.begin() + i * snapshot.current_len, snapshot.QRv.begin() + (i + 1) * snapshot.current_len);
        SigmaK(QKv_i, temp);
        energy_history[i] = -(ConvA(temp, QRv_i, snapshot.t1grid[i])[0] + Dflambda(QKv_i[0]) / snapshot.config_snapshot.T0);
    }
    
    // Save rvec history
    std::string rvecFilename = dirPath + "/rvec.txt";
    std::ofstream rvecFile(rvecFilename);
    if (rvecFile) {
        rvecFile << std::fixed << std::setprecision(16);
        rvecFile << "# Time\trvec\n";
        for (size_t i = 0; i < t1len; ++i) {
            rvecFile << snapshot.t1grid[i] << "\t" << snapshot.rvec[i] << "\n";
        }
        rvecFile.close();
    std::cout << dmfe::console::SAVE() << "Saved rvec history to " << rvecFilename << " (async)" << std::endl;
    } else {
    std::cerr << dmfe::console::ERR() << "Could not open file " << rvecFilename << std::endl;
    }
    
    // Save energy history
    std::string energyFilename = dirPath + "/energy.txt";
    std::ofstream energyFile(energyFilename);
    if (energyFile) {
        energyFile << std::fixed << std::setprecision(16);
        energyFile << "# Time\tEnergy\n";
        for (size_t i = 0; i < t1len; ++i) {
            energyFile << snapshot.t1grid[i] << "\t" << energy_history[i] << "\n";
        }
        energyFile.close();
    std::cout << dmfe::console::SAVE() << "Saved energy history to " << energyFilename << " (async)" << std::endl;
    } else {
    std::cerr << dmfe::console::ERR() << "Could not open file " << energyFilename << std::endl;
    }
    
    // Save QK[0] history
    std::string qk0Filename = dirPath + "/qk0.txt";
    std::ofstream qk0File(qk0Filename);
    if (qk0File) {
        qk0File << std::fixed << std::setprecision(16);
        qk0File << "# Time\tQK[0]\n";
        for (size_t i = 0; i < t1len; ++i) {
            qk0File << snapshot.t1grid[i] << "\t" << snapshot.QKv[i * snapshot.current_len] << "\n";
        }
        qk0File.close();
    std::cout << dmfe::console::SAVE() << "Saved QK[0] history to " << qk0Filename << " (async)" << std::endl;
    } else {
    std::cerr << dmfe::console::ERR() << "Could not open file " << qk0Filename << std::endl;
    }
    
    std::cout << dmfe::console::SAVE() << "Successfully saved complete history (" << t1len << " time points) to " << dirPath << " (async)" << std::endl;
}


