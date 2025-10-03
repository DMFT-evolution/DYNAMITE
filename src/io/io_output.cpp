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
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#if DMFE_WITH_CUDA
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#endif
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
void setupOutputDirectory()
{
    // Decide output root based on WHERE THE EXECUTABLE LIVES (not CWD)
    // This restores the original behavior requested by the user.
    char* homeDir = getenv("HOME");
    std::string exePath;
    {
        char exeBuf[PATH_MAX];
        ssize_t n = readlink("/proc/self/exe", exeBuf, sizeof(exeBuf) - 1);
        if (n > 0) { exeBuf[n] = '\0'; exePath = exeBuf; }
    }

    auto canonicalize = [](const std::string& p) -> std::string {
        char buf[PATH_MAX];
        if (!p.empty() && realpath(p.c_str(), buf)) return std::string(buf);
        return p;
    };

    std::string exeCanon = canonicalize(exePath);
    std::string homeCanon = (homeDir ? canonicalize(std::string(homeDir)) : std::string());
    if (!homeCanon.empty() && homeCanon.back() != '/') homeCanon += '/';

    if (!exeCanon.empty() && !homeCanon.empty()) {
        // If the executable resides under canonical HOME, force using outputDir
        if (exeCanon.rfind(homeCanon, 0) == 0) {
            config.resultsDir = config.outputDir;
        }
    }

    std::cout << "Executable (canonical): " << (exeCanon.empty() ? std::string("<unknown>") : exeCanon) << std::endl;
    std::cout << "HOME (canonical): " << (homeCanon.empty() ? std::string("<unknown>") : homeCanon) << std::endl;
    std::cout << "Selected results root: " << config.resultsDir << std::endl;
    
    // Check if directory already exists
    struct stat st = {0};
    if (stat(config.resultsDir.c_str(), &st) == 0) {
        // Directory exists
        if (S_ISDIR(st.st_mode)) {
            std::cout << "Directory already exists: " << config.resultsDir << std::endl;
            // Check if it's writable
            if (config.save_output && access(config.resultsDir.c_str(), W_OK) == 0) {
                std::cout << "Directory is writable." << std::endl;
                return; // Directory exists and is writable, no need to create it
            }
        }
    } else if (config.save_output) {
        // Directory doesn't exist, try to create it
        // For absolute paths, create all parent directories recursively
        if (config.resultsDir[0] == '/') {
            std::string path = "/";
            std::string dirPath = config.resultsDir.substr(1); // Remove leading '/'
            
            // Split the path and create each directory in sequence
            std::istringstream pathStream(dirPath);
            std::string dir;
            
            while (std::getline(pathStream, dir, '/')) {
                if (!dir.empty()) {
                    path += dir + "/";
                    if (stat(path.c_str(), &st) == -1) {
                        if (mkdir(path.c_str(), 0755) != 0) {
                            std::cerr << "Warning: Could not create directory " << path 
                                      << ": " << strerror(errno) << std::endl;
                            // Keep configured resultsDir as-is; later writes may fail, which is preferable to silently changing location
                            break;
                        }
                    }
                }
            }
        } else {
            // Relative path
            if (mkdir(config.resultsDir.c_str(), 0755) != 0) {
                std::cerr << "Warning: Could not create directory " << config.resultsDir 
                          << ": " << strerror(errno) << std::endl;
                // Keep configured resultsDir; do not silently change output location
            } else {
                std::cout << "Created output directory: " << config.resultsDir << std::endl;
            }
        }
    }
}

void saveParametersToFile(const std::string& dirPath, double delta, double delta_t)
{
    // Update peak memory before saving
    updatePeakMemory();
    
    std::string filename = dirPath + "/params.txt";
    std::ofstream params(filename);
    if (!params) {
        std::cerr << "Error: Could not open parameter file " << filename << std::endl;
        return;
    }
    
    // Calculate energy
    double energy;
#if DMFE_WITH_CUDA
    if (config.gpu) {
        energy = energyGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, config.T0);
    } else {
#endif
        vector<double> temp(config.len, 0.0);
        vector<double> lastQKv = getLastLenEntries(sim->h_QKv, config.len);
        SigmaK(lastQKv, temp);
        energy = -(ConvA(temp, getLastLenEntries(sim->h_QRv, config.len), sim->h_t1grid.back())[0] + Dflambda(lastQKv[0])/config.T0);
#if DMFE_WITH_CUDA
    }
#endif
    
    params << std::setprecision(16);
    params << "# DMFE Simulation Parameters" << std::endl;
    params << "# =========================" << std::endl;
    params << std::endl;

    // VERSION INFORMATION - Add this section first
    params << "# Version Information" << std::endl;
    params << "code_version = " << g_version_info.code_version << std::endl;
    params << "git_hash = " << g_version_info.git_hash << std::endl;
    params << "git_branch = " << g_version_info.git_branch << std::endl;
    if (g_version_info.git_tag != "unknown") {
        params << "git_tag = " << g_version_info.git_tag << std::endl;
    }
    params << "git_dirty = " << (g_version_info.git_dirty ? "true" : "false") << std::endl;
    params << "build_date = " << g_version_info.build_date << std::endl;
    params << "build_time = " << g_version_info.build_time << std::endl;
    params << "compiler_version = " << g_version_info.compiler_version << std::endl;
    params << "cuda_version = " << g_version_info.cuda_version << std::endl;
    params << std::endl;

    // Command Line Arguments
    params << "# Command Line Arguments" << std::endl;
    if (!config.command_line_args.empty()) {
        params << "command_line =";
        for (size_t i = 0; i < config.command_line_args.size(); ++i) {
            params << " " << config.command_line_args[i];
        }
        params << std::endl;
    } else {
        params << "command_line = (none stored)" << std::endl;
    }
    params << std::endl;

    // System and Performance Information
    params << "# System Information" << std::endl;
    params << "hostname = " << getHostname() << std::endl;
    params << "timestamp = " << getCurrentTimestamp() << std::endl;
    params << "config.gpu device = " << getGPUInfo() << std::endl;
    params << "execution mode = " << (config.gpu ? "GPU" : "CPU") << std::endl;
    params << std::endl;
    
    params << "# Performance Metrics" << std::endl;
    double runtime_seconds = getRuntimeSeconds();
    params << "runtime seconds = " << std::fixed << std::setprecision(2) << runtime_seconds << std::endl;
    params << "runtime formatted = " << formatDuration(runtime_seconds) << std::endl;
    params << "peak memory usage = " << formatMemory(peak_memory_kb) << std::endl;
    params << "peak memory (kb) = " << peak_memory_kb << std::endl;
    if (config.gpu) {
        params << "peak gpu memory (mb) = " << peak_gpu_memory_mb << std::endl;
        params << "current gpu memory (mb) = " << getGPUMemoryUsage() << std::endl;
    }
    params << "loops per second = " << std::fixed << std::setprecision(2) << (runtime_seconds > 0 ? config.loop / runtime_seconds : 0.0) << std::endl;
    params << std::setprecision(16) << std::defaultfloat;  // Reset again
    params << std::endl;
    
    params << "# Physical Parameters" << std::endl;
    params << "p = " << config.p << std::endl;
    params << "p2 = " << config.p2 << std::endl;
    params << "lambda = " << config.lambda << std::endl;
    // params << "TMCT = " << config.TMCT << std::endl;
    params << "T0 = " << config.T0 << std::endl;
    params << "Gamma = " << config.Gamma << std::endl;
    params << std::endl;
    
    params << "# Numerical Parameters" << std::endl;
    params << "len = " << config.len << std::endl;
    params << "tmax = " << config.tmax << std::endl;
    params << "delta_t_min = " << config.delta_t_min << std::endl;
    params << "delta_max = " << config.delta_max << std::endl;
    params << "use_serk2 = " << (config.use_serk2 ? "true" : "false") << std::endl;
    params << "aggressive_sparsify = " << (config.aggressive_sparsify ? "true" : "false") << std::endl;
    params << "maxLoop = " << config.maxLoop << std::endl;
    params << "rmax = [";
    for (size_t i = 0; i < config.rmax.size(); ++i) {
        params << config.rmax[i];
        if (i + 1 < config.rmax.size()) params << ", ";
    }
    params << "]" << std::endl;
    params << std::endl;
    
    params << "# Current Simulation State" << std::endl;
    params << "current_time = " << sim->h_t1grid.back() << std::endl;
    params << "current_loop = " << config.loop << std::endl;
    params << "current_delta = " << delta << std::endl;
    params << "current_delta_t = " << delta_t << std::endl;
    params << "current_method = " << (rk->init == 1 ? "RK54" : rk->init == 2 ? "SSPRK104" : "SERK2(" + std::to_string(2 * (rk->init - 2)) + ")") << std::endl;
    params << "current_t1grid_size = " << sim->h_t1grid.size() << std::endl;
    params << "current_QK0 = " << sim->h_QKv[(sim->h_t1grid.size() - 1) * config.len] << std::endl;
    params << "current_QR0 = " << sim->h_QRv[(sim->h_t1grid.size() - 1) * config.len] << std::endl;
    params << "current_r = " << sim->h_rvec.back() << std::endl;
    params << "current_energy = " << energy << std::endl;
    
    params.close();
    std::cout << "Saved parameters to " << filename << std::endl;
}

// Async version of saveParametersToFile that works with snapshot data
void saveParametersToFileAsync(const std::string& dirPath, double delta, double delta_t, const SimulationDataSnapshot& snapshot)
{
    std::string filename = dirPath + "/params.txt";
    std::ofstream params(filename);
    if (!params) {
        std::cerr << "Error: Could not open parameter file " << filename << std::endl;
        return;
    }
    
    params << std::setprecision(16);
    params << "# DMFE Simulation Parameters" << std::endl;
    params << "# =========================" << std::endl;
    params << std::endl;

    // VERSION INFORMATION - Add this section first
    params << "# Version Information" << std::endl;
    params << "code_version = " << snapshot.code_version << std::endl;
    params << "git_hash = " << snapshot.git_hash << std::endl;
    params << "git_branch = " << snapshot.git_branch << std::endl;
    if (snapshot.git_tag != "unknown") {
        params << "git_tag = " << snapshot.git_tag << std::endl;
    }
    params << "git_dirty = " << (snapshot.git_dirty ? "true" : "false") << std::endl;
    params << "build_date = " << snapshot.build_date << std::endl;
    params << "build_time = " << snapshot.build_time << std::endl;
    params << "compiler_version = " << snapshot.compiler_version << std::endl;
    params << "cuda_version = " << snapshot.cuda_version << std::endl;
    params << std::endl;

    // Command Line Arguments
    params << "# Command Line Arguments" << std::endl;
    if (!snapshot.config_snapshot.command_line_args.empty()) {
        params << "command_line =";
        for (size_t i = 0; i < snapshot.config_snapshot.command_line_args.size(); ++i) {
            params << " " << snapshot.config_snapshot.command_line_args[i];
        }
        params << std::endl;
    } else {
        params << "command_line = (none stored)" << std::endl;
    }
    params << std::endl;

    // System and Performance Information
    params << "# System Information" << std::endl;
    params << "hostname = " << getHostname() << std::endl;
    params << "timestamp = " << getCurrentTimestamp() << std::endl;
    params << "config.gpu device = " << getGPUInfo() << std::endl;
    params << "execution mode = " << (snapshot.config_snapshot.gpu ? "GPU" : "CPU") << std::endl;
    params << std::endl;
    
    params << "# Performance Metrics" << std::endl;
    auto now = std::chrono::system_clock::now();
    double runtime_seconds = std::chrono::duration<double>(now - snapshot.program_start_time_snapshot).count();
    params << "runtime seconds = " << std::fixed << std::setprecision(2) << runtime_seconds << std::endl;
    params << "runtime formatted = " << formatDuration(runtime_seconds) << std::endl;
    params << "peak memory usage = " << formatMemory(snapshot.peak_memory_kb_snapshot) << std::endl;
    params << "peak memory (kb) = " << snapshot.peak_memory_kb_snapshot << std::endl;
    if (snapshot.config_snapshot.gpu) {
        params << "peak gpu memory (mb) = " << snapshot.peak_gpu_memory_mb_snapshot << std::endl;
        // Note: We can't get current GPU memory in async context safely
        params << "current gpu memory (mb) = N/A (async)" << std::endl;
    }
    params << "loops per second = " << std::fixed << std::setprecision(2) << (runtime_seconds > 0 ? snapshot.current_loop / runtime_seconds : 0.0) << std::endl;
    params << std::setprecision(16) << std::defaultfloat;  // Reset again
    params << std::endl;
    
    params << "# Physical Parameters" << std::endl;
    params << "p = " << snapshot.config_snapshot.p << std::endl;
    params << "p2 = " << snapshot.config_snapshot.p2 << std::endl;
    params << "lambda = " << snapshot.config_snapshot.lambda << std::endl;
    params << "T0 = " << snapshot.config_snapshot.T0 << std::endl;
    params << "Gamma = " << snapshot.config_snapshot.Gamma << std::endl;
    params << std::endl;
    
    params << "# Numerical Parameters" << std::endl;
    params << "len = " << snapshot.config_snapshot.len << std::endl;
    params << "tmax = " << snapshot.config_snapshot.tmax << std::endl;
    params << "delta_t_min = " << snapshot.config_snapshot.delta_t_min << std::endl;
    params << "delta_max = " << snapshot.config_snapshot.delta_max << std::endl;
    params << "use_serk2 = " << (snapshot.config_snapshot.use_serk2 ? "true" : "false") << std::endl;
    params << "aggressive_sparsify = " << (snapshot.config_snapshot.aggressive_sparsify ? "true" : "false") << std::endl;
    params << "maxLoop = " << snapshot.config_snapshot.maxLoop << std::endl;
    params << "rmax = [";
    for (size_t i = 0; i < snapshot.config_snapshot.rmax.size(); ++i) {
        params << snapshot.config_snapshot.rmax[i];
        if (i + 1 < snapshot.config_snapshot.rmax.size()) params << ", ";
    }
    params << "]" << std::endl;
    params << std::endl;
    
    params << "# Current Simulation State" << std::endl;
    params << "current_time = " << snapshot.t_current << std::endl;
    params << "current_loop = " << snapshot.current_loop << std::endl;
    params << "current_delta = " << delta << std::endl;
    params << "current_delta_t = " << delta_t << std::endl;
    // Note: We can't access rk->init in async context, so we'll use a default
    params << "current_method = N/A (async)" << std::endl;
    params << "current_t1grid_size = " << snapshot.t1grid.size() << std::endl;
    params << "current_QK0 = " << snapshot.QKv[(snapshot.t1grid.size() - 1) * snapshot.current_len] << std::endl;
    params << "current_QR0 = " << snapshot.QRv[(snapshot.t1grid.size() - 1) * snapshot.current_len] << std::endl;
    params << "current_r = " << snapshot.rvec.back() << std::endl;
    params << "current_energy = " << snapshot.energy << std::endl;
    
    params.close();
    std::cout << "Saved parameters to " << filename << " (async)" << std::endl;
}

void saveSimulationStateBinary(const std::string& filename, double delta, double delta_t)
{
#if DMFE_WITH_CUDA
    // Copy data from GPU to CPU
    if (config.gpu) {
        copyVectorsToCPU(*sim);
    }
#endif

    // Calculate energy before saving
    double energy;
#if DMFE_WITH_CUDA
    if (config.gpu) {
        energy = energyGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, config.T0);
    } else {
#endif
        vector<double> temp(config.len, 0.0);
        vector<double> lastQKv = getLastLenEntries(sim->h_QKv, config.len);
        SigmaK(lastQKv, temp);
        energy = -(ConvA(temp, getLastLenEntries(sim->h_QRv, config.len), sim->h_t1grid.back())[0] + Dflambda(lastQKv[0])/config.T0);
#if DMFE_WITH_CUDA
    }
#endif
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    // Write header with metadata
    int header_version = 1;
    file.write(reinterpret_cast<char*>(&header_version), sizeof(int));
    
    // Write dimensions
    size_t t1grid_size = sim->h_t1grid.size();
    size_t vector_len = config.len;
    file.write(reinterpret_cast<char*>(&t1grid_size), sizeof(size_t));
    file.write(reinterpret_cast<char*>(&vector_len), sizeof(size_t));
    
    // Write parameters
    file.write(reinterpret_cast<const char*>(&config.p), sizeof(int));
    file.write(reinterpret_cast<const char*>(&config.p2), sizeof(int));
    file.write(reinterpret_cast<const char*>(&config.lambda), sizeof(double));
    file.write(reinterpret_cast<const char*>(&config.T0), sizeof(double));
    file.write(reinterpret_cast<const char*>(&config.Gamma), sizeof(double));
    file.write(reinterpret_cast<const char*>(&config.delta), sizeof(double));
    file.write(reinterpret_cast<const char*>(&config.delta_t), sizeof(double));
    file.write(reinterpret_cast<const char*>(&config.loop), sizeof(int));
    file.write(reinterpret_cast<const char*>(&energy), sizeof(double));
    
    // Write time grid
    file.write(reinterpret_cast<const char*>(sim->h_t1grid.data()), sim->h_t1grid.size() * sizeof(double));
    
    // Write vectors
    file.write(reinterpret_cast<const char*>(sim->h_QKv.data()), sim->h_QKv.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(sim->h_QRv.data()), sim->h_QRv.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(sim->h_dQKv.data()), sim->h_dQKv.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(sim->h_dQRv.data()), sim->h_dQRv.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(sim->h_rvec.data()), sim->h_rvec.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(sim->h_drvec.data()), sim->h_drvec.size() * sizeof(double));
    
    std::cout << "Saved binary data to " << filename << std::endl;
    
    // Also write parameters to a separate text file
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    saveParametersToFile(dirPath, delta, delta_t);

#if DMFE_WITH_CUDA
    saveHistory(filename, delta, delta_t, *sim, config.len, config.T0, config.gpu);
#endif
}

#if defined(H5_RUNTIME_OPTIONAL) || defined(USE_HDF5)
void saveSimulationStateHDF5(const std::string& filename, double delta, double delta_t)
{
#if defined(H5_RUNTIME_OPTIONAL)
    if (!h5rt::available()) throw std::runtime_error("HDF5 not available at runtime");
#if DMFE_WITH_CUDA
    // Ensure data on CPU
    if (config.gpu) {
        copyVectorsToCPU(*sim);
    }
#endif
    // Calculate energy
    double energy;
#if DMFE_WITH_CUDA
    if (config.gpu) {
        energy = energyGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, config.T0);
    } else {
#endif
        vector<double> temp(config.len, 0.0);
        vector<double> lastQKv = getLastLenEntries(sim->h_QKv, config.len);
        SigmaK(lastQKv, temp);
        energy = -(ConvA(temp, getLastLenEntries(sim->h_QRv, config.len), sim->h_t1grid.back())[0] + Dflambda(lastQKv[0])/config.T0);
#if DMFE_WITH_CUDA
    }
#endif
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    std::string binFilename = dirPath + "/data.bin";
    if (fileExists(binFilename)) { std::remove(binFilename.c_str()); }

    auto file = h5rt::create_file_trunc(filename.c_str());
    if (file < 0) throw std::runtime_error("Failed to create HDF5 file");

    auto fail_and_fallback = [&](const char* why){
        std::cerr << "[HDF5] write failed: " << why << "; falling back to binary." << std::endl;
        h5rt::close_file(file);
        // Remove possibly empty/partial file
        std::remove(filename.c_str());
        saveSimulationStateBinary(dirPath + "/data.bin", delta, delta_t);
    };

    // Datasets (fail fast if any write fails)
    if (!h5rt::write_dataset_1d_double(file, "QKv", sim->h_QKv.data(), sim->h_QKv.size())) { fail_and_fallback("QKv"); return; }
    if (!h5rt::write_dataset_1d_double(file, "QRv", sim->h_QRv.data(), sim->h_QRv.size())) { fail_and_fallback("QRv"); return; }
    if (!h5rt::write_dataset_1d_double(file, "dQKv", sim->h_dQKv.data(), sim->h_dQKv.size())) { fail_and_fallback("dQKv"); return; }
    if (!h5rt::write_dataset_1d_double(file, "dQRv", sim->h_dQRv.data(), sim->h_dQRv.size())) { fail_and_fallback("dQRv"); return; }
    if (!h5rt::write_dataset_1d_double(file, "t1grid", sim->h_t1grid.data(), sim->h_t1grid.size())) { fail_and_fallback("t1grid"); return; }
    if (!h5rt::write_dataset_1d_double(file, "rvec", sim->h_rvec.data(), sim->h_rvec.size())) { fail_and_fallback("rvec"); return; }
    if (!h5rt::write_dataset_1d_double(file, "drvec", sim->h_drvec.data(), sim->h_drvec.size())) { fail_and_fallback("drvec"); return; }

    // Attributes
    double t_current = sim->h_t1grid.back();
    int current_len = config.len; int current_loop = config.loop;
    if (!h5rt::write_attr_double(file, "time", t_current)) { fail_and_fallback("attr time"); return; }
    if (!h5rt::write_attr_int(file, "iteration", current_loop)) { fail_and_fallback("attr iteration"); return; }
    if (!h5rt::write_attr_int(file, "len", current_len)) { fail_and_fallback("attr len"); return; }
    if (!h5rt::write_attr_double(file, "delta", config.delta)) { fail_and_fallback("attr delta"); return; }
    if (!h5rt::write_attr_double(file, "delta_t", config.delta_t)) { fail_and_fallback("attr delta_t"); return; }
    if (!h5rt::write_attr_double(file, "T0", config.T0)) { fail_and_fallback("attr T0"); return; }
    if (!h5rt::write_attr_double(file, "lambda", config.lambda)) { fail_and_fallback("attr lambda"); return; }
    if (!h5rt::write_attr_int(file, "p", config.p)) { fail_and_fallback("attr p"); return; }
    if (!h5rt::write_attr_int(file, "p2", config.p2)) { fail_and_fallback("attr p2"); return; }
    if (!h5rt::write_attr_double(file, "Gamma", config.Gamma)) { fail_and_fallback("attr Gamma"); return; }
    if (!h5rt::write_attr_double(file, "delta_t_min", config.delta_t_min)) { fail_and_fallback("attr delta_t_min"); return; }
    if (!h5rt::write_attr_double(file, "delta_max", config.delta_max)) { fail_and_fallback("attr delta_max"); return; }
    if (!h5rt::write_attr_int(file, "use_serk2", config.use_serk2 ? 1 : 0)) { fail_and_fallback("attr use_serk2"); return; }
    if (!h5rt::write_attr_int(file, "aggressive_sparsify", config.aggressive_sparsify ? 1 : 0)) { fail_and_fallback("attr aggressive_sparsify"); return; }
    if (!h5rt::write_attr_double(file, "energy", energy)) { fail_and_fallback("attr energy"); return; }

    h5rt::close_file(file);
    std::cout << "Saved HDF5 data to " << filename << std::endl;
    saveParametersToFile(dirPath, delta, delta_t);
#if DMFE_WITH_CUDA
    saveHistory(filename, delta, delta_t, *sim, config.len, config.T0, config.gpu);
#endif
#elif defined(USE_HDF5)
#if DMFE_WITH_CUDA
    // Copy data from GPU to CPU
    copyVectorsToCPU(*sim);
#endif
    
    // Calculate energy before saving
    double energy;
#if DMFE_WITH_CUDA
    if (config.gpu) {
        energy = energyGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, config.T0);
    } else {
#endif
        vector<double> temp(config.len, 0.0);
        vector<double> lastQKv = getLastLenEntries(sim->h_QKv, config.len);
        SigmaK(lastQKv, temp);
        energy = -(ConvA(temp, getLastLenEntries(sim->h_QRv, config.len), sim->h_t1grid.back())[0] + Dflambda(lastQKv[0])/config.T0);
#if DMFE_WITH_CUDA
    }
#endif
    
    // Get directory path from filename
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    
    // Remove any existing binary file to avoid confusion
    std::string binFilename = dirPath + "/data.bin";
    if (fileExists(binFilename)) {
        std::remove(binFilename.c_str());
        std::cout << "Removed existing binary file: " << binFilename << std::endl;
    }
    
    // Create HDF5 file
    H5::H5File file(filename, H5F_ACC_TRUNC);
    
    // Base compression settings
    H5::DSetCreatPropList plist;
    plist.setDeflate(6); // Compression level
    plist.setShuffle();
    
    // Write QKv dataset with appropriate chunking
    {
        hsize_t qkv_dims[1] = {sim->h_QKv.size()};
        H5::DataSpace qkv_space(1, qkv_dims);
        
        // Set chunk size specifically for this dataset
        size_t chunk_size = std::min(size_t(1048576), sim->h_QKv.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        
        file.createDataSet("QKv", H5::PredType::NATIVE_DOUBLE, qkv_space, plist)
            .write(sim->h_QKv.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    // Write QRv dataset with appropriate chunking
    {
        hsize_t qrv_dims[1] = {sim->h_QRv.size()};
        H5::DataSpace qrv_space(1, qrv_dims);
        
        // Set chunk size specifically for this dataset
        size_t chunk_size = std::min(size_t(1048576), sim->h_QRv.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        
        file.createDataSet("QRv", H5::PredType::NATIVE_DOUBLE, qrv_space, plist)
            .write(sim->h_QRv.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    // Write dQKv dataset with appropriate chunking
    {
        hsize_t dqkv_dims[1] = {sim->h_dQKv.size()};
        H5::DataSpace dqkv_space(1, dqkv_dims);
        
        // Set chunk size specifically for this dataset
        size_t chunk_size = std::min(size_t(1048576), sim->h_dQKv.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        
        file.createDataSet("dQKv", H5::PredType::NATIVE_DOUBLE, dqkv_space, plist)
            .write(sim->h_dQKv.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    // Write dQRv dataset with appropriate chunking
    {
        hsize_t dqrv_dims[1] = {sim->h_dQRv.size()};
        H5::DataSpace dqrv_space(1, dqrv_dims);
        
        // Set chunk size specifically for this dataset
        size_t chunk_size = std::min(size_t(1048576), sim->h_dQRv.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        
        file.createDataSet("dQRv", H5::PredType::NATIVE_DOUBLE, dqrv_space, plist)
            .write(sim->h_dQRv.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    // Write t1grid dataset with appropriate chunking
    {
        hsize_t t1_dims[1] = {sim->h_t1grid.size()};
        H5::DataSpace t1_space(1, t1_dims);
        
        // Set chunk size specifically for this dataset - important for smaller arrays!
        size_t chunk_size = std::min(size_t(1024), sim->h_t1grid.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        
        file.createDataSet("t1grid", H5::PredType::NATIVE_DOUBLE, t1_space, plist)
            .write(sim->h_t1grid.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    // Write rvec dataset with appropriate chunking
    {
        hsize_t r_dims[1] = {sim->h_rvec.size()};
        H5::DataSpace r_space(1, r_dims);
        
        // Set chunk size specifically for this dataset - important for smaller arrays!
        size_t chunk_size = std::min(size_t(1024), sim->h_rvec.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        
        file.createDataSet("rvec", H5::PredType::NATIVE_DOUBLE, r_space, plist)
            .write(sim->h_rvec.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    // Write drvec dataset with appropriate chunking
    {
        hsize_t dr_dims[1] = {sim->h_drvec.size()};
        H5::DataSpace dr_space(1, dr_dims);
        
        // Set chunk size specifically for this dataset - important for smaller arrays!
        size_t chunk_size = std::min(size_t(1024), sim->h_drvec.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        
        file.createDataSet("drvec", H5::PredType::NATIVE_DOUBLE, dr_space, plist)
            .write(sim->h_drvec.data(), H5::PredType::NATIVE_DOUBLE);
    }

    // Add metadata as attributes
    H5::DataSpace scalar_space(H5S_SCALAR);
    H5::DataType string_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);
    
    auto add_string_attr = [&file, &scalar_space, &string_type](const char* name, 
                                                               const std::string& value) {
        const char* str_data = value.c_str();
        file.createAttribute(name, string_type, scalar_space)
            .write(string_type, &str_data);
    };
    
    auto add_attr = [&file, &scalar_space](const char* name, 
                                          const H5::PredType& type, 
                                          const void* value) {
        file.createAttribute(name, type, scalar_space)
            .write(type, value);
    };

    // Simulation parameters
    double t_current = sim->h_t1grid.back();
    int current_len = config.len;
    int current_loop = config.loop;

    // Version attributes
    add_string_attr("code_version", g_version_info.code_version);
    add_string_attr("git_hash", g_version_info.git_hash);
    add_string_attr("git_branch", g_version_info.git_branch);
    add_string_attr("build_date", g_version_info.build_date + " " + g_version_info.build_time);
    add_string_attr("compiler", g_version_info.compiler_version);
    add_string_attr("cuda_version", g_version_info.cuda_version);
    
    add_attr("time", H5::PredType::NATIVE_DOUBLE, &t_current);
    add_attr("iteration", H5::PredType::NATIVE_INT, &current_loop);
    add_attr("len", H5::PredType::NATIVE_INT, &current_len);
    add_attr("delta", H5::PredType::NATIVE_DOUBLE, &config.delta);
    add_attr("delta_t", H5::PredType::NATIVE_DOUBLE, &config.delta_t);
    add_attr("T0", H5::PredType::NATIVE_DOUBLE, &config.T0);
    add_attr("lambda", H5::PredType::NATIVE_DOUBLE, &config.lambda);
    add_attr("p", H5::PredType::NATIVE_INT, &config.p);
    add_attr("p2", H5::PredType::NATIVE_INT, &config.p2);
    add_attr("Gamma", H5::PredType::NATIVE_DOUBLE, &config.Gamma);
    add_attr("delta_t_min", H5::PredType::NATIVE_DOUBLE, &config.delta_t_min);
    add_attr("delta_max", H5::PredType::NATIVE_DOUBLE, &config.delta_max);
    int use_serk2_int = config.use_serk2 ? 1 : 0;
    add_attr("use_serk2", H5::PredType::NATIVE_INT, &use_serk2_int);
    int aggressive_sparsify_int = config.aggressive_sparsify ? 1 : 0;
    add_attr("aggressive_sparsify", H5::PredType::NATIVE_INT, &aggressive_sparsify_int);
    add_attr("energy", H5::PredType::NATIVE_DOUBLE, &energy);
    
    std::cout << "Saved HDF5 data to " << filename 
              << " (time=" << t_current 
              << ", vectors=" << sim->h_QKv.size() / config.len << "×" << config.len 
              << ", energy=" << energy
              << ")" << std::endl;

    // Save parameter text file directly
    saveParametersToFile(dirPath, delta, delta_t);

#if DMFE_WITH_CUDA
    saveHistory(filename, delta, delta_t, *sim, config.len, config.T0, config.gpu);
#endif
#endif // inner: H5_RUNTIME_OPTIONAL or USE_HDF5
}
#endif // defined(H5_RUNTIME_OPTIONAL) || defined(USE_HDF5)

#if defined(H5_RUNTIME_OPTIONAL) || defined(USE_HDF5)
void saveSimulationStateHDF5Async(const std::string& filename, const SimulationDataSnapshot& snapshot)
{
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    std::string binFilename = dirPath + "/data.bin";
    if (fileExists(binFilename)) { std::remove(binFilename.c_str()); }

#if defined(H5_RUNTIME_OPTIONAL)
    if (!h5rt::available()) throw std::runtime_error("HDF5 not available at runtime");
    // Use the snapshot data directly
    auto file = h5rt::create_file_trunc(filename.c_str());
    if (file < 0) throw std::runtime_error("Failed to create HDF5 file");

    auto fail_and_fallback = [&](const char* why){
        std::cerr << "[HDF5] write failed: " << why << "; falling back to binary." << std::endl;
        h5rt::close_file(file);
        // Remove possibly empty/partial file
        std::remove(filename.c_str());
        throw std::runtime_error(std::string("HDF5 write failed: ") + why);
    };

    // Datasets (fail fast if any write fails)
    if (!h5rt::write_dataset_1d_double(file, "QKv", snapshot.QKv.data(), snapshot.QKv.size())) { fail_and_fallback("QKv"); }
    if (!h5rt::write_dataset_1d_double(file, "QRv", snapshot.QRv.data(), snapshot.QRv.size())) { fail_and_fallback("QRv"); }
    if (!h5rt::write_dataset_1d_double(file, "dQKv", snapshot.dQKv.data(), snapshot.dQKv.size())) { fail_and_fallback("dQKv"); }
    if (!h5rt::write_dataset_1d_double(file, "dQRv", snapshot.dQRv.data(), snapshot.dQRv.size())) { fail_and_fallback("dQRv"); }
    if (!h5rt::write_dataset_1d_double(file, "t1grid", snapshot.t1grid.data(), snapshot.t1grid.size())) { fail_and_fallback("t1grid"); }
    if (!h5rt::write_dataset_1d_double(file, "rvec", snapshot.rvec.data(), snapshot.rvec.size())) { fail_and_fallback("rvec"); }
    if (!h5rt::write_dataset_1d_double(file, "drvec", snapshot.drvec.data(), snapshot.drvec.size())) { fail_and_fallback("drvec"); }

    // Attributes
    if (!h5rt::write_attr_double(file, "time", snapshot.t_current)) { fail_and_fallback("attr time"); }
    if (!h5rt::write_attr_int(file, "iteration", snapshot.current_loop)) { fail_and_fallback("attr iteration"); }
    if (!h5rt::write_attr_int(file, "len", snapshot.current_len)) { fail_and_fallback("attr len"); }
    if (!h5rt::write_attr_double(file, "delta", snapshot.config_snapshot.delta)) { fail_and_fallback("attr delta"); }
    if (!h5rt::write_attr_double(file, "delta_t", snapshot.config_snapshot.delta_t)) { fail_and_fallback("attr delta_t"); }
    if (!h5rt::write_attr_double(file, "T0", snapshot.config_snapshot.T0)) { fail_and_fallback("attr T0"); }
    if (!h5rt::write_attr_double(file, "lambda", snapshot.config_snapshot.lambda)) { fail_and_fallback("attr lambda"); }
    if (!h5rt::write_attr_int(file, "p", snapshot.config_snapshot.p)) { fail_and_fallback("attr p"); }
    if (!h5rt::write_attr_int(file, "p2", snapshot.config_snapshot.p2)) { fail_and_fallback("attr p2"); }
    if (!h5rt::write_attr_double(file, "Gamma", snapshot.config_snapshot.Gamma)) { fail_and_fallback("attr Gamma"); }
    if (!h5rt::write_attr_double(file, "delta_t_min", snapshot.config_snapshot.delta_t_min)) { fail_and_fallback("attr delta_t_min"); }
    if (!h5rt::write_attr_double(file, "delta_max", snapshot.config_snapshot.delta_max)) { fail_and_fallback("attr delta_max"); }
    if (!h5rt::write_attr_int(file, "use_serk2", snapshot.config_snapshot.use_serk2 ? 1 : 0)) { fail_and_fallback("attr use_serk2"); }
    if (!h5rt::write_attr_int(file, "aggressive_sparsify", snapshot.config_snapshot.aggressive_sparsify ? 1 : 0)) { fail_and_fallback("attr aggressive_sparsify"); }
    if (!h5rt::write_attr_double(file, "energy", snapshot.energy)) { fail_and_fallback("attr energy"); }

    h5rt::close_file(file);
    std::cout << "Saved HDF5 data to " << filename << " (async)" << std::endl;
    
    // For async, we need to reconstruct the SimulationData for saveHistory
    // This is a bit hacky, but necessary since saveHistory expects the global sim
    // We'll create a temporary SimulationData with the snapshot data
    SimulationData temp_sim;
    temp_sim.h_QKv = snapshot.QKv;
    temp_sim.h_QRv = snapshot.QRv;
    temp_sim.h_dQKv = snapshot.dQKv;
    temp_sim.h_dQRv = snapshot.dQRv;
    temp_sim.h_t1grid = snapshot.t1grid;
    temp_sim.h_rvec = snapshot.rvec;
    temp_sim.h_drvec = snapshot.drvec;
    
    saveParametersToFileAsync(dirPath, snapshot.config_snapshot.delta, snapshot.config_snapshot.delta_t, snapshot);
#if DMFE_WITH_CUDA
    saveHistoryAsync(filename, snapshot.config_snapshot.delta, snapshot.config_snapshot.delta_t, snapshot);
#endif

#elif defined(USE_HDF5)
    // Create HDF5 file
    H5::H5File file(filename, H5F_ACC_TRUNC);
    
    // Base compression settings
    H5::DSetCreatPropList plist;
    plist.setDeflate(6); // Compression level
    plist.setShuffle();
    
    // Write datasets using snapshot data
    {
        hsize_t qkv_dims[1] = {snapshot.QKv.size()};
        H5::DataSpace qkv_space(1, qkv_dims);
        size_t chunk_size = std::min(size_t(1048576), snapshot.QKv.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        file.createDataSet("QKv", H5::PredType::NATIVE_DOUBLE, qkv_space, plist)
            .write(snapshot.QKv.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    {
        hsize_t qrv_dims[1] = {snapshot.QRv.size()};
        H5::DataSpace qrv_space(1, qrv_dims);
        size_t chunk_size = std::min(size_t(1048576), snapshot.QRv.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        file.createDataSet("QRv", H5::PredType::NATIVE_DOUBLE, qrv_space, plist)
            .write(snapshot.QRv.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    {
        hsize_t dqkv_dims[1] = {snapshot.dQKv.size()};
        H5::DataSpace dqkv_space(1, dqkv_dims);
        size_t chunk_size = std::min(size_t(1048576), snapshot.dQKv.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        file.createDataSet("dQKv", H5::PredType::NATIVE_DOUBLE, dqkv_space, plist)
            .write(snapshot.dQKv.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    {
        hsize_t dqrv_dims[1] = {snapshot.dQRv.size()};
        H5::DataSpace dqrv_space(1, dqrv_dims);
        size_t chunk_size = std::min(size_t(1048576), snapshot.dQRv.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        file.createDataSet("dQRv", H5::PredType::NATIVE_DOUBLE, dqrv_space, plist)
            .write(snapshot.dQRv.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    {
        hsize_t t1_dims[1] = {snapshot.t1grid.size()};
        H5::DataSpace t1_space(1, t1_dims);
        size_t chunk_size = std::min(size_t(1024), snapshot.t1grid.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        file.createDataSet("t1grid", H5::PredType::NATIVE_DOUBLE, t1_space, plist)
            .write(snapshot.t1grid.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    {
        hsize_t r_dims[1] = {snapshot.rvec.size()};
        H5::DataSpace r_space(1, r_dims);
        size_t chunk_size = std::min(size_t(1024), snapshot.rvec.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        file.createDataSet("rvec", H5::PredType::NATIVE_DOUBLE, r_space, plist)
            .write(snapshot.rvec.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    {
        hsize_t dr_dims[1] = {snapshot.drvec.size()};
        H5::DataSpace dr_space(1, dr_dims);
        size_t chunk_size = std::min(size_t(1024), snapshot.drvec.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        file.createDataSet("drvec", H5::PredType::NATIVE_DOUBLE, dr_space, plist)
            .write(snapshot.drvec.data(), H5::PredType::NATIVE_DOUBLE);
    }

    // Add metadata as attributes
    H5::DataSpace scalar_space(H5S_SCALAR);
    H5::DataType string_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);
    
    auto add_string_attr = [&file, &scalar_space, &string_type](const char* name, 
                                                               const std::string& value) {
        const char* str_data = value.c_str();
        file.createAttribute(name, string_type, scalar_space)
            .write(string_type, &str_data);
    };
    
    auto add_attr = [&file, &scalar_space](const char* name, 
                                          const H5::PredType& type, 
                                          const void* value) {
        file.createAttribute(name, type, scalar_space)
            .write(type, value);
    };

    // Version attributes
    add_string_attr("code_version", g_version_info.code_version);
    add_string_attr("git_hash", g_version_info.git_hash);
    add_string_attr("git_branch", g_version_info.git_branch);
    add_string_attr("build_date", g_version_info.build_date + " " + g_version_info.build_time);
    add_string_attr("compiler", g_version_info.compiler_version);
    add_string_attr("cuda_version", g_version_info.cuda_version);
    
    add_attr("time", H5::PredType::NATIVE_DOUBLE, &snapshot.t_current);
    add_attr("iteration", H5::PredType::NATIVE_INT, &snapshot.current_loop);
    add_attr("len", H5::PredType::NATIVE_INT, &snapshot.current_len);
    add_attr("delta", H5::PredType::NATIVE_DOUBLE, &snapshot.config_snapshot.delta);
    add_attr("delta_t", H5::PredType::NATIVE_DOUBLE, &snapshot.config_snapshot.delta_t);
    add_attr("T0", H5::PredType::NATIVE_DOUBLE, &snapshot.config_snapshot.T0);
    add_attr("lambda", H5::PredType::NATIVE_DOUBLE, &snapshot.config_snapshot.lambda);
    add_attr("p", H5::PredType::NATIVE_INT, &snapshot.config_snapshot.p);
    add_attr("p2", H5::PredType::NATIVE_INT, &snapshot.config_snapshot.p2);
    add_attr("Gamma", H5::PredType::NATIVE_DOUBLE, &snapshot.config_snapshot.Gamma);
    add_attr("delta_t_min", H5::PredType::NATIVE_DOUBLE, &snapshot.config_snapshot.delta_t_min);
    add_attr("delta_max", H5::PredType::NATIVE_DOUBLE, &snapshot.config_snapshot.delta_max);
    int use_serk2_int = snapshot.config_snapshot.use_serk2 ? 1 : 0;
    add_attr("use_serk2", H5::PredType::NATIVE_INT, &use_serk2_int);
    int aggressive_sparsify_int = snapshot.config_snapshot.aggressive_sparsify ? 1 : 0;
    add_attr("aggressive_sparsify", H5::PredType::NATIVE_INT, &aggressive_sparsify_int);
    add_attr("energy", H5::PredType::NATIVE_DOUBLE, &snapshot.energy);
    
    std::cout << "Saved HDF5 data to " << filename 
              << " (time=" << snapshot.t_current 
              << ", vectors=" << snapshot.QKv.size() / snapshot.current_len << "×" << snapshot.current_len 
              << ", energy=" << snapshot.energy
              << ") (async)" << std::endl;

    // For async, we need to reconstruct the SimulationData for saveHistory
    SimulationData temp_sim;
    temp_sim.h_QKv = snapshot.QKv;
    temp_sim.h_QRv = snapshot.QRv;
    temp_sim.h_dQKv = snapshot.dQKv;
    temp_sim.h_dQRv = snapshot.dQRv;
    temp_sim.h_t1grid = snapshot.t1grid;
    temp_sim.h_rvec = snapshot.rvec;
    temp_sim.h_drvec = snapshot.drvec;
    
    saveParametersToFileAsync(dirPath, snapshot.config_snapshot.delta, snapshot.config_snapshot.delta_t, snapshot);
    saveHistoryAsync(filename, snapshot.config_snapshot.delta, snapshot.config_snapshot.delta_t, snapshot);
    saveCompressedDataAsync(dirPath, snapshot);
#endif // H5_RUNTIME_OPTIONAL or USE_HDF5
}
#endif // defined(H5_RUNTIME_OPTIONAL) || defined(USE_HDF5)

SimulationDataSnapshot saveSimulationState(const std::string& filename, double delta, double delta_t)
{
    // Function to save simulation state, either synchronously or asynchronously
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    
    // Wait for any ongoing save to complete
    {
        std::cout << "Waiting for ongoing save to complete..." << std::endl;
        std::unique_lock<std::mutex> lock(saveMutex);
        saveCondition.wait(lock, []{ return !saveInProgress; });
        saveInProgress = true;
        std::cout << "Save lock acquired, proceeding with snapshot creation" << std::endl;
    }
    
#if DMFE_WITH_CUDA
    // Create snapshot synchronously to pause simulation during this critical section
    SimulationDataSnapshot snapshot = createDataSnapshot();
#else
    // Create empty snapshot for CPU-only builds
    SimulationDataSnapshot snapshot;
    std::cerr << "Warning: Snapshot creation not available in CPU-only builds" << std::endl;
#endif
    
    // Check if async export is enabled
    if (config.async_export) {
        // For asynchronous saving, use the snapshot in background thread for I/O operations
        auto saveAsync = [filename, dirPath, delta, delta_t, snapshot]() {
            std::cout << "Background save thread started for " << filename << std::endl;
            try {
#if defined(H5_RUNTIME_OPTIONAL)
                if (h5rt::available()) {
                    saveSimulationStateHDF5Async(filename, snapshot);
                    saveCompressedDataAsync(dirPath, snapshot);
                } else {
                    // Fallback to binary
                    saveSimulationStateBinary(dirPath + "/data.bin", delta, delta_t);
                    saveCompressedDataAsync(dirPath, snapshot);
                }
#elif defined(USE_HDF5)
                saveSimulationStateHDF5Async(filename, snapshot);
                saveCompressedDataAsync(dirPath, snapshot);
#else
                saveSimulationStateBinary(dirPath + "/data.bin", delta, delta_t);
                saveCompressedDataAsync(dirPath, snapshot);
#endif
            } catch (const std::exception& e) {
                std::cerr << "Error in background save: " << e.what() << std::endl;
                // Fallback to synchronous binary save
                try {
                    saveSimulationStateBinary(dirPath + "/data.bin", delta, delta_t);
                    saveCompressedDataAsync(dirPath, snapshot);
                } catch (...) {
                    std::cerr << "Fallback save also failed" << std::endl;
                }
            }
            
            // Signal that save is complete
            {
                std::lock_guard<std::mutex> lock(saveMutex);
                saveInProgress = false;
                std::cout << "Background save completed for " << filename << ", saveInProgress set to false" << std::endl;
            }
            saveCondition.notify_one();
        };
        
        // Launch the save operation in a background thread
        std::thread saveThread(saveAsync);
        saveThread.detach();
        
        std::cout << "Created snapshot and started background save to " << filename << std::endl;
    } else {
        // Synchronous saving mode
        std::cout << "Performing synchronous save to " << filename << std::endl;
        try {
#if defined(H5_RUNTIME_OPTIONAL)
            if (h5rt::available()) {
                saveSimulationStateHDF5Async(filename, snapshot);
                saveCompressedDataAsync(dirPath, snapshot);
            } else {
                // Fallback to binary
                saveSimulationStateBinary(dirPath + "/data.bin", delta, delta_t);
                saveCompressedDataAsync(dirPath, snapshot);
            }
#elif defined(USE_HDF5)
            saveSimulationStateHDF5Async(filename, snapshot);
            saveCompressedDataAsync(dirPath, snapshot);
#else
            saveSimulationStateBinary(dirPath + "/data.bin", delta, delta_t);
            saveCompressedDataAsync(dirPath, snapshot);
#endif
        } catch (const std::exception& e) {
            std::cerr << "Error in synchronous save: " << e.what() << std::endl;
            // Fallback to synchronous binary save
            try {
                saveSimulationStateBinary(dirPath + "/data.bin", delta, delta_t);
                saveCompressedDataAsync(dirPath, snapshot);
            } catch (...) {
                std::cerr << "Fallback save also failed" << std::endl;
            }
        }
        
        // Signal that save is complete
        {
            std::lock_guard<std::mutex> lock(saveMutex);
            saveInProgress = false;
            std::cout << "Synchronous save completed for " << filename << ", saveInProgress set to false" << std::endl;
        }
        saveCondition.notify_one();
    }
    
    // Return the snapshot for consistent final output
    return snapshot;
}

void saveCompressedData(const std::string& dirPath)
{
    // Ensure data is on CPU if using GPU
#if DMFE_WITH_CUDA
    if (config.gpu) {
        sim->h_QKB1int.resize(sim->d_QKB1int.size());
        thrust::copy(sim->d_QKB1int.begin(), sim->d_QKB1int.end(), sim->h_QKB1int.begin());
        sim->h_QRB1int.resize(sim->d_QRB1int.size());
        thrust::copy(sim->d_QRB1int.begin(), sim->d_QRB1int.end(), sim->h_QRB1int.begin());
        sim->h_theta.resize(sim->d_theta.size());
        thrust::copy(sim->d_theta.begin(), sim->d_theta.end(), sim->h_theta.begin());
        sim->h_t1grid.resize(sim->d_t1grid.size());
        thrust::copy(sim->d_t1grid.begin(), sim->d_t1grid.end(), sim->h_t1grid.begin());
    }
#endif
    
    // Save QKB1int to QK_compressed
    std::string qk_filename = dirPath + "/QK_compressed";
    std::ofstream qk_file(qk_filename, std::ios::binary);
    if (!qk_file) {
        std::cerr << "Error: Could not open file " << qk_filename << std::endl;
        return;
    }
    
    // Write header (dimensions information)
    size_t rows = config.len;
    size_t cols = config.len;
    qk_file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    qk_file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    
    // Write data
    qk_file.write(reinterpret_cast<const char*>(sim->h_QKB1int.data()), sim->h_QKB1int.size() * sizeof(double));
    qk_file.close();
    
    // Save QRB1int to QR_compressed
    std::string qr_filename = dirPath + "/QR_compressed";
    std::ofstream qr_file(qr_filename, std::ios::binary);
    if (!qr_file) {
        std::cerr << "Error: Could not open file " << qr_filename << std::endl;
        return;
    }
    
    // Write header (dimensions information)
    qr_file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    qr_file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    
    // Write data
    qr_file.write(reinterpret_cast<const char*>(sim->h_QRB1int.data()), sim->h_QRB1int.size() * sizeof(double));
    qr_file.close();
    
    // Save (last t1) * theta to t1_compressed.txt
    double last_t1 = sim->h_t1grid.back();
    std::string t1_filename = dirPath + "/t1_compressed.txt";
    std::ofstream t1_file(t1_filename);
    if (!t1_file) {
        std::cerr << "Error: Could not open file " << t1_filename << std::endl;
        return;
    }
    
    t1_file << std::fixed << std::setprecision(16);
    for (size_t i = 0; i < sim->h_theta.size(); ++i) {
        t1_file << last_t1 * sim->h_theta[i] << "\n";
    }
    t1_file.close();
    
    std::cout << "Saved compressed data to " << qk_filename << ", " << qr_filename << ", and " << t1_filename << std::endl;
}

// Async version of saveCompressedData that works with snapshot data
void saveCompressedDataAsync(const std::string& dirPath, const SimulationDataSnapshot& snapshot)
{
    // Save QKB1int to QK_compressed
    std::string qk_filename = dirPath + "/QK_compressed";
    std::ofstream qk_file(qk_filename, std::ios::binary);
    if (!qk_file) {
        std::cerr << "Error: Could not open file " << qk_filename << std::endl;
        return;
    }
    
    // Write header (dimensions information)
    size_t rows = snapshot.current_len;
    size_t cols = snapshot.current_len;
    qk_file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    qk_file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    
    // Write data
    qk_file.write(reinterpret_cast<const char*>(snapshot.QKB1int.data()), snapshot.QKB1int.size() * sizeof(double));
    qk_file.close();
    
    // Save QRB1int to QR_compressed
    std::string qr_filename = dirPath + "/QR_compressed";
    std::ofstream qr_file(qr_filename, std::ios::binary);
    if (!qr_file) {
        std::cerr << "Error: Could not open file " << qr_filename << std::endl;
        return;
    }
    
    // Write header (dimensions information)
    qr_file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    qr_file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    
    // Write data
    qr_file.write(reinterpret_cast<const char*>(snapshot.QRB1int.data()), snapshot.QRB1int.size() * sizeof(double));
    qr_file.close();
    
    // Save (last t1) * theta to t1_compressed.txt
    double last_t1 = snapshot.t1grid.back();
    std::string t1_filename = dirPath + "/t1_compressed.txt";
    std::ofstream t1_file(t1_filename);
    if (!t1_file) {
        std::cerr << "Error: Could not open file " << t1_filename << std::endl;
        return;
    }
    
    t1_file << std::fixed << std::setprecision(16);
    for (size_t i = 0; i < snapshot.theta.size(); ++i) {
        t1_file << last_t1 * snapshot.theta[i] << "\n";
    }
    t1_file.close();
    
    std::cout << "Saved compressed data to " << qk_filename << ", " << qr_filename << ", and " << t1_filename << " (async)" << std::endl;
}
