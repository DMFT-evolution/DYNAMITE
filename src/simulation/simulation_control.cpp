#include "simulation/simulation_control.hpp"
#include "simulation/simulation_data.hpp"
#if DMFE_WITH_CUDA
#include "simulation/device_simulation_data.hpp"
#endif
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
        auto& times = sim->host->debug_step_times;
        auto& runtimes = sim->host->debug_step_runtimes;
        auto& memory = sim->host->debug_step_memory;

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
    size_t currentSize = config.gpu ? sim->device->t1grid.size() : sim->host->t1grid.size();
    
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
        sim->device->t1grid.resize(targetSize);
        sim->device->delta_t_ratio.resize(targetSize);
        sim->device->QKv.resize(targetSize * config.len);
        sim->device->QRv.resize(targetSize * config.len);
        sim->device->dQKv.resize(targetSize * config.len);
        sim->device->dQRv.resize(targetSize * config.len);
        sim->device->rvec.resize(targetSize);
        sim->device->drvec.resize(targetSize);
        
        // Update simulation state variables
        config.delta_t = sim->device->t1grid[targetSize-1] - sim->device->t1grid[targetSize-2];
        config.loop -= n;

        interpolateGPU();
    } else {
        // Resize host (CPU) vectors to target size via SimulationData
        sim->host->t1grid.resize(targetSize);
        sim->host->delta_t_ratio.resize(targetSize);
        sim->host->QKv.resize(targetSize * config.len);
        sim->host->QRv.resize(targetSize * config.len);
        sim->host->dQKv.resize(targetSize * config.len);
        sim->host->dQRv.resize(targetSize * config.len);
        sim->host->rvec.resize(targetSize);
        sim->host->drvec.resize(targetSize);

        // Update simulation state variables
        config.delta_t = sim->host->t1grid[targetSize-1] - sim->host->t1grid[targetSize-2];
        config.loop -= n;

        interpolate();
    }
    trim_debug_timelines_by_n(static_cast<size_t>(n));
    
    std::cout << dmfe::console::INFO() << "Successfully rolled back " << n
              << " iterations to time t = "
              << (config.gpu ? sim->device->t1grid.back() : sim->host->t1grid.back()) << std::endl;
#else
    // Debug telemetry is not sparsified/downsampled; rollbacks must trim by *n*, not by t1grid size.
    auto trim_debug_timelines_by_n = [&](size_t n_trim) {
        auto& times = sim->host->debug_step_times;
        auto& runtimes = sim->host->debug_step_runtimes;
        auto& memory = sim->host->debug_step_memory;

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
    size_t currentSize = sim->host->t1grid.size();
    
    // Check if we have enough history to roll back
    if (n >= currentSize - 1) {
        std::cerr << dmfe::console::ERR() << "Cannot roll back " << n << " iterations. Only "
                  << (currentSize - 1) << " iterations available." << std::endl;
        return false;
    }
    
    // Calculate target size
    size_t targetSize = currentSize - n;
    
    // Resize host (CPU) vectors to target size via SimulationData
    sim->host->t1grid.resize(targetSize);
    sim->host->delta_t_ratio.resize(targetSize);
    sim->host->QKv.resize(targetSize * config.len);
    sim->host->QRv.resize(targetSize * config.len);
    sim->host->dQKv.resize(targetSize * config.len);
    sim->host->dQRv.resize(targetSize * config.len);
    sim->host->rvec.resize(targetSize);
    sim->host->drvec.resize(targetSize);

    // Update simulation state variables
    config.delta_t = sim->host->t1grid[targetSize-1] - sim->host->t1grid[targetSize-2];
    config.loop -= n;

    interpolate();
    trim_debug_timelines_by_n(static_cast<size_t>(n));
    
    std::cout << dmfe::console::INFO() << "Successfully rolled back " << n
              << " iterations to time t = "
              << sim->host->t1grid.back() << std::endl;
#endif
    return true;
}
