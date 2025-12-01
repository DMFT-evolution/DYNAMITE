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
#include <unordered_map>

#include "core/console.hpp"
#include "version/version_compat.hpp"

using namespace std;

extern SimulationConfig config;
extern SimulationData* sim;
extern RKData* rk;
extern size_t peak_memory_kb;
extern size_t peak_gpu_memory_mb;
extern std::chrono::high_resolution_clock::time_point program_start_time;

// Local helper: read grid generation params from Grid_data/<len>/grid_params.txt into a simple dictionary
static bool readGridParamsForLen(std::size_t len, std::unordered_map<std::string, std::string>& out)
{
    out.clear();
    std::ostringstream p;
    p << "Grid_data/" << len << "/grid_params.txt";
    std::ifstream ifs(p.str());
    if (!ifs) return false;
    std::string line;
    auto trim = [](const std::string& s) {
        auto a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) return std::string();
        auto b = s.find_last_not_of(" \t\r\n");
        return s.substr(a, b - a + 1);
    };
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (!key.empty()) out[key] = val;
    }
    return !out.empty();
}

void saveParametersToFile(const std::string& dirPath, double delta, double delta_t)
{
    // Update peak memory before saving
    updatePeakMemory();

    std::string filename = dirPath + "/params.txt";
    std::ofstream params(filename);
    if (!params) {
        std::cerr << dmfe::console::ERR() << "Could not open parameter file " << filename << std::endl;
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

    // Progress mapping for params write: [0.50 .. 0.65]
    const double p_start_params = 0.50;
    const double p_end_params   = 0.65;
    const double p_span_params  = (p_end_params - p_start_params);
    const double last_t1_params = sim->h_t1grid.empty() ? 0.0 : sim->h_t1grid.back();
    auto update_params_prog = [&](int step, int total_steps){
        if (total_steps <= 0) return;
        double frac = p_start_params + p_span_params * std::min(std::max(0, step), total_steps) / (double)total_steps;
        _setSaveProgress(frac, last_t1_params, "params");
    };
    int params_steps_total = 8; // rough sections below
    int params_step = 0;
    update_params_prog(params_step, params_steps_total);

    params << std::setprecision(16);
    params << "# DYNAMITE Simulation Parameters" << std::endl;
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
    update_params_prog(++params_step, params_steps_total);

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
    update_params_prog(++params_step, params_steps_total);

    // System and Performance Information
    params << "# System Information" << std::endl;
    params << "hostname = " << getHostname() << std::endl;
    params << "timestamp = " << getCurrentTimestamp() << std::endl;
    params << "config.gpu device = " << getGPUInfo() << std::endl;
    params << "execution mode = " << (config.gpu ? "GPU" : "CPU") << std::endl;
    params << std::endl;
    update_params_prog(++params_step, params_steps_total);

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
    update_params_prog(++params_step, params_steps_total);

    params << "# Physical Parameters" << std::endl;
    params << "p = " << config.p << std::endl;
    params << "p2 = " << config.p2 << std::endl;
    params << "lambda = " << config.lambda << std::endl;
    params << "T0 = " << config.T0 << std::endl;
    params << "Gamma = " << config.Gamma << std::endl;
    params << std::endl;
    update_params_prog(++params_step, params_steps_total);

    params << "# Numerical Parameters" << std::endl;
    params << "len = " << config.len << std::endl;
    params << "tmax = " << config.tmax << std::endl;
    params << "delta_t_min = " << config.delta_t_min << std::endl;
    params << "delta_max = " << config.delta_max << std::endl;
    params << "use_serk2 = " << (config.use_serk2 ? "true" : "false") << std::endl;
    // Interpolation mode for QR/dQR
    params << "log_response_interp = " << (config.log_response_interp ? "true" : "false") << std::endl;
    // Tail-fit stabilization toggle
    params << "tail_fit_enabled = " << (config.tail_fit_enabled ? "true" : "false") << std::endl;
    params << "sparsify_sweeps = " << config.sparsify_sweeps << std::endl;
    params << "maxLoop = " << config.maxLoop << std::endl;
    params << "rmax = [";
    for (size_t i = 0; i < config.rmax.size(); ++i) {
        params << config.rmax[i];
        if (i + 1 < config.rmax.size()) params << ", ";
    }
    params << "]" << std::endl;
    params << std::endl;

    // Grid Generation Parameters (provenance for Grid_data used)
    {
        std::unordered_map<std::string, std::string> gp;
        if (readGridParamsForLen(config.len, gp)) {
            params << "# Grid Generation Parameters" << std::endl;
            auto writeIfRaw = [&](const char* k, const char* outKey){ auto it = gp.find(k); if (it != gp.end()) params << outKey << " = " << it->second << std::endl; };
            auto writeIfDouble = [&](const char* k, const char* outKey){ auto it = gp.find(k); if (it != gp.end()) { try { double v = std::stod(it->second); params << outKey << " = " << std::setprecision(16) << std::defaultfloat << v << std::endl; } catch(...) { params << outKey << " = " << it->second << std::endl; } } };
            writeIfRaw("subdir", "grid_subdir");
            writeIfRaw("len", "grid_len");
            // Force double-precision formatting for these numeric fields
            writeIfDouble("Tmax", "grid_Tmax");
            writeIfRaw("spline_order", "grid_spline_order");
            writeIfRaw("interp_method", "grid_interp_method");
            writeIfRaw("interp_order", "grid_interp_order");
            writeIfRaw("fh_stencil", "grid_fh_stencil");
            writeIfDouble("alpha", "grid_alpha");
            writeIfDouble("delta", "grid_delta");
            params << std::endl;
            update_params_prog(++params_step, params_steps_total);
        } else {
            params << "# Grid Generation Parameters" << std::endl;
            params << "# grid_params.txt not found for len=" << config.len << std::endl;
            params << std::endl;
            update_params_prog(++params_step, params_steps_total);
        }
    }

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
    if (config.debug) {
        dmfe::console::end_status_line_if_needed(dmfe::console::stdout_is_tty());
        invalidateStatusAnchor();
        std::cout << dmfe::console::SAVE() << "Saved parameters to " << filename << std::endl;
    }
    update_params_prog(params_steps_total, params_steps_total);
}

void saveParametersToFileAsync(const std::string& dirPath, double delta, double delta_t, const SimulationDataSnapshot& snapshot)
{
    std::string filename = dirPath + "/params.txt";
    std::ofstream params(filename);
    if (!params) {
        std::cerr << dmfe::console::ERR() << "Could not open parameter file " << filename << std::endl;
        return;
    }

    // Progress mapping for params write (async): [0.50 .. 0.65]
    const double p_start_params = 0.50;
    const double p_end_params   = 0.65;
    const double p_span_params  = (p_end_params - p_start_params);
    const double last_t1_params = snapshot.t1grid.empty() ? 0.0 : snapshot.t1grid.back();
    auto update_params_prog = [&](int step, int total_steps){
        if (total_steps <= 0) return;
        double frac = p_start_params + p_span_params * std::min(std::max(0, step), total_steps) / (double)total_steps;
        _setSaveProgress(frac, last_t1_params, "params");
    };
    int params_steps_total = 8;
    int params_step = 0;
    update_params_prog(params_step, params_steps_total);

    params << std::setprecision(16);
    params << "# DYNAMITE Simulation Parameters" << std::endl;
    params << "# =========================" << std::endl;
    params << std::endl;
    update_params_prog(++params_step, params_steps_total);

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
    update_params_prog(++params_step, params_steps_total);

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
    update_params_prog(++params_step, params_steps_total);

    // System and Performance Information
    params << "# System Information" << std::endl;
    params << "hostname = " << getHostname() << std::endl;
    params << "timestamp = " << getCurrentTimestamp() << std::endl;
    params << "config.gpu device = " << getGPUInfo() << std::endl;
    params << "execution mode = " << (snapshot.config_snapshot.gpu ? "GPU" : "CPU") << std::endl;
    params << std::endl;
    update_params_prog(++params_step, params_steps_total);

    params << "# Performance Metrics" << std::endl;
    auto now = std::chrono::system_clock::now();
    double runtime_seconds = std::chrono::duration<double>(now - snapshot.program_start_time_snapshot).count();
    params << "runtime seconds = " << std::fixed << std::setprecision(2) << runtime_seconds << std::endl;
    params << "runtime formatted = " << formatDuration(runtime_seconds) << std::endl;
    params << "peak memory usage = " << formatMemory(snapshot.peak_memory_kb_snapshot) << std::endl;
    params << "peak memory (kb) = " << snapshot.peak_memory_kb_snapshot << std::endl;
    if (snapshot.config_snapshot.gpu) {
        params << "peak gpu memory (mb) = " << snapshot.peak_gpu_memory_mb_snapshot << std::endl;
        params << "current gpu memory (mb) = N/A (async)" << std::endl;
    }
    params << "loops per second = " << std::fixed << std::setprecision(2) << (runtime_seconds > 0 ? snapshot.current_loop / runtime_seconds : 0.0) << std::endl;
    params << std::setprecision(16) << std::defaultfloat;  // Reset again
    params << std::endl;
    update_params_prog(++params_step, params_steps_total);

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
    // Interpolation mode for QR/dQR
    params << "log_response_interp = " << (snapshot.config_snapshot.log_response_interp ? "true" : "false") << std::endl;
    // Tail-fit stabilization toggle
    params << "tail_fit_enabled = " << (snapshot.config_snapshot.tail_fit_enabled ? "true" : "false") << std::endl;
    params << "sparsify_sweeps = " << snapshot.config_snapshot.sparsify_sweeps << std::endl;
    params << "maxLoop = " << snapshot.config_snapshot.maxLoop << std::endl;
    params << "rmax = [";
    for (size_t i = 0; i < snapshot.config_snapshot.rmax.size(); ++i) {
        params << snapshot.config_snapshot.rmax[i];
        if (i + 1 < snapshot.config_snapshot.rmax.size()) params << ", ";
    }
    params << "]" << std::endl;
    params << std::endl;

    // Grid Generation Parameters (async snapshot)
    {
        std::unordered_map<std::string, std::string> gp;
        if (readGridParamsForLen(snapshot.current_len, gp)) {
            params << "# Grid Generation Parameters" << std::endl;
            auto writeIfRaw = [&](const char* k, const char* outKey){ auto it = gp.find(k); if (it != gp.end()) params << outKey << " = " << it->second << std::endl; };
            auto writeIfDouble = [&](const char* k, const char* outKey){ auto it = gp.find(k); if (it != gp.end()) { try { double v = std::stod(it->second); params << outKey << " = " << std::setprecision(16) << std::defaultfloat << v << std::endl; } catch(...) { params << outKey << " = " << it->second << std::endl; } } };
            writeIfRaw("subdir", "grid_subdir");
            writeIfRaw("len", "grid_len");
            writeIfDouble("Tmax", "grid_Tmax");
            writeIfRaw("spline_order", "grid_spline_order");
            writeIfRaw("interp_method", "grid_interp_method");
            writeIfRaw("interp_order", "grid_interp_order");
            writeIfRaw("fh_stencil", "grid_fh_stencil");
            writeIfDouble("alpha", "grid_alpha");
            writeIfDouble("delta", "grid_delta");
            params << std::endl;
            update_params_prog(++params_step, params_steps_total);
        } else {
            params << "# Grid Generation Parameters" << std::endl;
            params << "# grid_params.txt not found for len=" << snapshot.current_len << std::endl;
            params << std::endl;
            update_params_prog(++params_step, params_steps_total);
        }
    }

    params << "# Current Simulation State" << std::endl;
    params << "current_time = " << snapshot.t_current << std::endl;
    params << "current_loop = " << snapshot.current_loop << std::endl;
    params << "current_delta = " << delta << std::endl;
    params << "current_delta_t = " << delta_t << std::endl;
    params << "current_method = N/A (async)" << std::endl;
    params << "current_t1grid_size = " << snapshot.t1grid.size() << std::endl;
    params << "current_QK0 = " << snapshot.QKv[(snapshot.t1grid.size() - 1) * snapshot.current_len] << std::endl;
    params << "current_QR0 = " << snapshot.QRv[(snapshot.t1grid.size() - 1) * snapshot.current_len] << std::endl;
    params << "current_r = " << snapshot.rvec.back() << std::endl;
    params << "current_energy = " << snapshot.energy << std::endl;

    params.close();
    update_params_prog(params_steps_total, params_steps_total);
    if (config.debug) {
        dmfe::console::end_status_line_if_needed(dmfe::console::stdout_is_tty());
        invalidateStatusAnchor();
        std::cout << dmfe::console::SAVE() << "Saved parameters to " << filename << " (async)" << std::endl;
    }
}
