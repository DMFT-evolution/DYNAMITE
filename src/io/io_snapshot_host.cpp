#if !DMFE_WITH_CUDA

#include "io/io_utils.hpp"
#include "simulation/simulation_data.hpp"
#include "core/config.hpp"
#include "math/math_sigma.hpp"
#include "math/math_ops.hpp"
#include "convolution/convolution.hpp"
#include "EOMs/time_steps.hpp"
#include "version/version_info.hpp"
#include <vector>

extern SimulationConfig config;
extern SimulationData* sim;
extern size_t peak_memory_kb;
extern size_t peak_gpu_memory_mb;
extern std::chrono::high_resolution_clock::time_point program_start_time;

SimulationDataSnapshot createDataSnapshot() {
    SimulationDataSnapshot snapshot;

    snapshot.QKv = sim->h_QKv;
    snapshot.QRv = sim->h_QRv;
    snapshot.dQKv = sim->h_dQKv;
    snapshot.dQRv = sim->h_dQRv;
    snapshot.t1grid = sim->h_t1grid;
    snapshot.rvec = sim->h_rvec;
    snapshot.drvec = sim->h_drvec;
    snapshot.QKB1int = sim->h_QKB1int;
    snapshot.QRB1int = sim->h_QRB1int;
    snapshot.theta = sim->h_theta;
    snapshot.t_current = sim->h_t1grid.empty() ? 0.0 : sim->h_t1grid.back();

    std::vector<double> temp(config.len, 0.0);
    std::vector<double> lastQKv = getLastLenEntries(sim->h_QKv, config.len);
    SigmaK(lastQKv, temp);
    snapshot.energy = -(ConvA(temp, getLastLenEntries(sim->h_QRv, config.len), sim->h_t1grid.back())[0]
                        + Dflambda(lastQKv[0]) / config.T0);

    snapshot.current_len = config.len;
    snapshot.current_loop = config.loop;
    snapshot.config_snapshot = config;

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

    snapshot.peak_memory_kb_snapshot = peak_memory_kb;
    snapshot.peak_gpu_memory_mb_snapshot = peak_gpu_memory_mb;
    snapshot.program_start_time_snapshot = program_start_time;

    snapshot.debug_step_times = sim->h_debug_step_times;
    snapshot.debug_step_runtimes = sim->h_debug_step_runtimes;

    return snapshot;
}

#endif // !DMFE_WITH_CUDA
