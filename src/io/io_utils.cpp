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
std::mutex saveMutex;
bool saveInProgress = false;
std::condition_variable saveCondition;

// Save telemetry state (protected by saveMutex)
static SaveTelemetry g_saveTelemetry{};
static std::atomic<bool> g_saveTelemetryDirty{false};
static std::atomic<bool> g_statusAnchorInvalidated{false};

SaveTelemetry getSaveTelemetry() {
    std::lock_guard<std::mutex> lock(saveMutex);
    return g_saveTelemetry;
}

void _setSaveStart(const std::string& filename) {
    std::lock_guard<std::mutex> lock(saveMutex);
    g_saveTelemetry.in_progress = true;
    g_saveTelemetry.target_file = filename;
    g_saveTelemetry.last_start_time = std::chrono::high_resolution_clock::now();
    g_saveTelemetryDirty.store(true, std::memory_order_relaxed);
    g_statusAnchorInvalidated.store(true, std::memory_order_relaxed);
    // Print on its own line so TUI can redraw cleanly on next tick
    dmfe::console::end_status_line_if_needed(dmfe::console::stdout_is_tty());
    std::cout << dmfe::console::SAVE() << "Save started: " << filename << std::endl << std::flush;
}

void _setSaveEnd(const std::string& filename) {
    std::lock_guard<std::mutex> lock(saveMutex);
    g_saveTelemetry.in_progress = false;
    g_saveTelemetry.last_completed_file = filename;
    g_saveTelemetry.last_end_time = std::chrono::high_resolution_clock::now();
    g_saveTelemetryDirty.store(true, std::memory_order_relaxed);
    g_statusAnchorInvalidated.store(true, std::memory_order_relaxed);
    dmfe::console::end_status_line_if_needed(dmfe::console::stdout_is_tty());
    std::cout << dmfe::console::DONE() << "Save finished: " << filename << std::endl << std::flush;
}

void markSaveTelemetryDirty() {
    g_saveTelemetryDirty.store(true, std::memory_order_relaxed);
}

bool consumeSaveTelemetryDirty() {
    return g_saveTelemetryDirty.exchange(false, std::memory_order_acq_rel);
}

void invalidateStatusAnchor() {
    g_statusAnchorInvalidated.store(true, std::memory_order_relaxed);
}

bool consumeStatusAnchorInvalidated() {
    return g_statusAnchorInvalidated.exchange(false, std::memory_order_acq_rel);
}

// Helper function to find an existing parameter directory with matching parameters
std::string findExistingParamDir(const std::string& resultsDir_param, int p_param, int p2_param,
                                double lambda_param, double T0_param, double Gamma_param, size_t len_param,
                                double delta_t_min_param, double delta_max_param, bool use_serk2_param, bool aggressive_sparsify_param) {
    std::vector<std::string> dirs_to_check = {"Results/"};
    std::string resultsDir = resultsDir_param;
    if (!resultsDir.empty() && resultsDir.back() != '/') {
        resultsDir += '/';
    }
    if (resultsDir != "Results/") {
        dirs_to_check.push_back(resultsDir);
    }

    for (const auto& dir : dirs_to_check) {
        DIR* d = opendir(dir.c_str());
        if (!d) continue;

        struct dirent* entry;
        while ((entry = readdir(d)) != nullptr) {
            if (entry->d_type == DT_DIR && entry->d_name[0] != '.') {
                std::string subdir = dir + entry->d_name;
                std::string param_file = subdir + "/params.txt";
                if (fileExists(param_file) && checkParametersMatch(param_file, p_param, p2_param, lambda_param, T0_param, Gamma_param, len_param, delta_t_min_param, delta_max_param, use_serk2_param, aggressive_sparsify_param)) {
                    // Also check version compatibility
                    VersionAnalysis analysis = analyzeVersionCompatibility(param_file);
                    if (analysis.level == VersionCompatibility::IDENTICAL || analysis.level == VersionCompatibility::COMPATIBLE || analysis.level == VersionCompatibility::WARNING) {
                        closedir(d);
                        return subdir;
                    }
                }
            }
        }
        closedir(d);
    }
    return "";
}

std::string getParameterDirPath(const std::string& resultsDir_param, int p_param, int p2_param,
                               double lambda_param, double T0_param, double Gamma_param, size_t len_param)
{
    std::time_t now = std::time(nullptr);
    std::tm* tm = std::localtime(&now);
    int day = tm->tm_mday;
    int month = tm->tm_mon + 1;
    int year = tm->tm_year + 1900;
    int hour = tm->tm_hour;
    int min = tm->tm_min;
    std::ostringstream d;
    d << resultsDir_param;
    if (!resultsDir_param.empty() && resultsDir_param.back() != '/') {
        d << "/";
    }
    d << std::setfill('0') << std::setw(2) << day << "_"
      << std::setw(2) << month << "_"
      << year << "_"
      << std::setw(2) << hour << std::setw(2) << min << "_"
      << "p=" << p_param << "_q=" << p2_param << "_lambda=" << std::fixed << std::setprecision(2) << lambda_param
      << "_T0=";
    if (T0_param >= 1e50) {
        d << "inf";
    } else {
        d << std::fixed << std::setprecision(5) << T0_param;
    }
    d << "_G=" << std::fixed << std::setprecision(2) << Gamma_param << "_L=" << len_param;
    return d.str();
}

void ensureDirectoryExists(const std::string &dir)
{
    struct stat st{};
    if (stat(dir.c_str(), &st) == -1)
    {
        std::string cmd = "mkdir -p '" + dir + "'";
        if (system(cmd.c_str()) != 0)
            std::cerr << "Warning: Could not create directory " << dir << std::endl;
    }
}

std::string getFilename(const std::string& resultsDir_param, int p_param, int p2_param,
                       double lambda_param, double T0_param, double Gamma_param, size_t len_param,
                       double delta_t_min_param, double delta_max_param, bool use_serk2_param, bool aggressive_sparsify_param,
                       bool save_output_param)
{
    // First, try to find an existing directory with matching parameters
    std::string existing_dir = findExistingParamDir(resultsDir_param, p_param, p2_param, lambda_param, T0_param, Gamma_param, len_param, delta_t_min_param, delta_max_param, use_serk2_param, aggressive_sparsify_param);
    if (!existing_dir.empty()) {
        return existing_dir + "/data.h5";
    }

    // No existing directory found, create a new one
    std::string base_path = getParameterDirPath(resultsDir_param, p_param, p2_param, lambda_param, T0_param, Gamma_param, len_param);
    std::string path = base_path;
    int suffix = 0;
    while (true) {
        std::string param_file = path + "/params.txt";
        if (!fileExists(param_file)) {
            // no params, assume ok
            break;
        }
        if (checkParametersMatch(param_file, p_param, p2_param, lambda_param, T0_param, Gamma_param, len_param, delta_t_min_param, delta_max_param, use_serk2_param, aggressive_sparsify_param)) {
            // match, ok
            break;
        }
        // conflict, append suffix to time
        suffix++;
        size_t p_pos = path.find("_p=");
        if (p_pos == std::string::npos) break; // error
        std::string prefix = path.substr(0, p_pos);
        size_t last_ = prefix.rfind('_');
        if (last_ == std::string::npos) break;
        std::string time_part = prefix.substr(last_ + 1);
        std::string new_time = time_part + "-" + std::to_string(suffix);
        path = prefix.substr(0, last_ + 1) + new_time + path.substr(p_pos);
    }
    if (save_output_param)
        ensureDirectoryExists(path);
    return path + "/data.h5";
}

bool fileExists(const std::string &filename)
{
    std::ifstream f(filename);
    return f.good();
}

// Utility functions for save operations
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

double getRuntimeSeconds() {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - program_start_time);
    return duration.count() / 1000.0;
}

std::string formatDuration(double total_seconds) {
    int days = static_cast<int>(total_seconds / 86400);
    int hours = static_cast<int>((total_seconds - days * 86400) / 3600);
    int minutes = static_cast<int>((total_seconds - days * 86400 - hours * 3600) / 60);
    double seconds = total_seconds - days * 86400 - hours * 3600 - minutes * 60;

    std::ostringstream oss;
    if (days > 0) {
        oss << days << "d " << hours << "h " << minutes << "m " << std::fixed << std::setprecision(1) << seconds << "s";
    } else if (hours > 0) {
        oss << hours << "h " << minutes << "m " << std::fixed << std::setprecision(1) << seconds << "s";
    } else if (minutes > 0) {
        oss << minutes << "m " << std::fixed << std::setprecision(1) << seconds << "s";
    } else {
        oss << std::fixed << std::setprecision(2) << seconds << "s";
    }
    return oss.str();
}

std::string getGPUInfo() {
#if DMFE_WITH_CUDA
    if (!config.gpu) return "None (CPU only)";

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) return "None detected";

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::ostringstream oss;
    oss << prop.name << " (Compute " << prop.major << "." << prop.minor << ")";
    return oss.str();
#else
    return "None (CPU only - built without CUDA)";
#endif
}

std::string formatMemory(size_t memory_kb) {
    if (memory_kb >= 1024 * 1024) {
        return std::to_string(memory_kb / (1024 * 1024)) + " GB";
    } else if (memory_kb >= 1024) {
        return std::to_string(memory_kb / 1024) + " MB";
    } else {
        return std::to_string(memory_kb) + " KB";
    }
}

// Function to wait for any ongoing async saves to complete
void waitForAsyncSavesToComplete() {
    std::cout << dmfe::console::INFO() << "Waiting for any ongoing async saves to complete..." << std::endl;
    std::unique_lock<std::mutex> lock(saveMutex);
    saveCondition.wait(lock, []{ return !saveInProgress; });
    std::cout << dmfe::console::DONE() << "All async saves completed." << std::endl;
}
