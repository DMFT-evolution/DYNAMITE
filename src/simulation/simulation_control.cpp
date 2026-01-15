#include "simulation/simulation_control.hpp"
#include "simulation/simulation_data.hpp"
#include "core/config.hpp"
#include "core/config_build.hpp"
#include "core/console.hpp"
#include "interpolation/interpolation_core.hpp"
#include <iostream>
#include <algorithm>

// External global variables
extern SimulationConfig config;
extern SimulationData* sim;

bool rollbackState(int n) {
#if DMFE_WITH_CUDA
    // Debug telemetry is not sparsified/downsampled; rollbacks must trim by *n*, not by t1grid size.
    auto trim_debug_timelines_by_n = [&](size_t n_trim) {
        auto& times = sim->h_debug_step_times;
        auto& runtimes = sim->h_debug_step_runtimes;
        auto& memory = sim->h_debug_step_memory;

        // Bring vectors into a consistent state (best-effort) before trimming.
        const size_t common_before = std::min(times.size(), std::min(runtimes.size(), memory.size()));
        if (times.size() != common_before) times.resize(common_before);
        if (runtimes.size() != common_before) runtimes.resize(common_before);
        if (memory.size() != common_before) memory.resize(common_before);

        if (common_before == 0) return;

        const size_t n_effective = std::min(n_trim, common_before);
        const size_t common_after = common_before - n_effective;
        times.resize(common_after);
        runtimes.resize(common_after);
        memory.resize(common_after);
    };

    // Get current state size
    size_t currentSize = config.gpu ? sim->d_t1grid.size() : sim->h_t1grid.size();
    
    // Check if we have enough history to roll back
    if (n >= currentSize - 1) {
        std::cerr << dmfe::console::ERR() << "Cannot roll back " << n << " iterations. Only "
                  << (currentSize - 1) << " iterations available." << std::endl;
        return false;
    }
    
    // Calculate target size
    size_t targetSize = currentSize - n;
    
    if (config.gpu) {
        // Resize GPU vectors to target size
        sim->d_t1grid.resize(targetSize);
        sim->d_delta_t_ratio.resize(targetSize);
        sim->d_QKv.resize(targetSize * config.len);
        sim->d_QRv.resize(targetSize * config.len);
        sim->d_dQKv.resize(targetSize * config.len);
        sim->d_dQRv.resize(targetSize * config.len);
        sim->d_rvec.resize(targetSize);
        sim->d_drvec.resize(targetSize);
        
        // Update simulation state variables
        config.delta_t = sim->d_t1grid[targetSize-1] - sim->d_t1grid[targetSize-2];
        config.loop -= n;

        interpolateGPU();
    } else {
        // Resize host (CPU) vectors to target size via SimulationData
        sim->h_t1grid.resize(targetSize);
        sim->h_delta_t_ratio.resize(targetSize);
        sim->h_QKv.resize(targetSize * config.len);
        sim->h_QRv.resize(targetSize * config.len);
        sim->h_dQKv.resize(targetSize * config.len);
        sim->h_dQRv.resize(targetSize * config.len);
        sim->h_rvec.resize(targetSize);
        sim->h_drvec.resize(targetSize);

        // Update simulation state variables
        config.delta_t = sim->h_t1grid[targetSize-1] - sim->h_t1grid[targetSize-2];
        config.loop -= n;

        interpolate();
    }
    trim_debug_timelines_by_n(static_cast<size_t>(n));
    
    std::cout << dmfe::console::INFO() << "Successfully rolled back " << n
              << " iterations to time t = "
              << (config.gpu ? sim->d_t1grid.back() : sim->h_t1grid.back()) << std::endl;
#else
    // Debug telemetry is not sparsified/downsampled; rollbacks must trim by *n*, not by t1grid size.
    auto trim_debug_timelines_by_n = [&](size_t n_trim) {
        auto& times = sim->h_debug_step_times;
        auto& runtimes = sim->h_debug_step_runtimes;
        auto& memory = sim->h_debug_step_memory;

        const size_t common_before = std::min(times.size(), std::min(runtimes.size(), memory.size()));
        if (times.size() != common_before) times.resize(common_before);
        if (runtimes.size() != common_before) runtimes.resize(common_before);
        if (memory.size() != common_before) memory.resize(common_before);

        if (common_before == 0) return;

        const size_t n_effective = std::min(n_trim, common_before);
        const size_t common_after = common_before - n_effective;
        times.resize(common_after);
        runtimes.resize(common_after);
        memory.resize(common_after);
    };

    // Get current state size
    size_t currentSize = sim->h_t1grid.size();
    
    // Check if we have enough history to roll back
    if (n >= currentSize - 1) {
        std::cerr << dmfe::console::ERR() << "Cannot roll back " << n << " iterations. Only "
                  << (currentSize - 1) << " iterations available." << std::endl;
        return false;
    }
    
    // Calculate target size
    size_t targetSize = currentSize - n;
    
    // Resize host (CPU) vectors to target size via SimulationData
    sim->h_t1grid.resize(targetSize);
    sim->h_delta_t_ratio.resize(targetSize);
    sim->h_QKv.resize(targetSize * config.len);
    sim->h_QRv.resize(targetSize * config.len);
    sim->h_dQKv.resize(targetSize * config.len);
    sim->h_dQRv.resize(targetSize * config.len);
    sim->h_rvec.resize(targetSize);
    sim->h_drvec.resize(targetSize);

    // Update simulation state variables
    config.delta_t = sim->h_t1grid[targetSize-1] - sim->h_t1grid[targetSize-2];
    config.loop -= n;

    interpolate();
    trim_debug_timelines_by_n(static_cast<size_t>(n));
    
    std::cout << dmfe::console::INFO() << "Successfully rolled back " << n
              << " iterations to time t = "
              << sim->h_t1grid.back() << std::endl;
#endif
    return true;
}
