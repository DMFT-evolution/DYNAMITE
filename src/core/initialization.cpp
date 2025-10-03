#include "core/initialization.hpp"
#include "core/config.hpp"
#include "core/config_build.hpp"
#include "simulation/simulation_data.hpp"
#include "EOMs/rk_data.hpp"
#if DMFE_WITH_CUDA
#include "core/device_utils.cuh"
#include "core/gpu_memory_utils.hpp"
#endif
#include "io/io_utils.hpp"
#include "math/math_ops.hpp"
#include "EOMs/time_steps.hpp"
#include "EOMs/runge_kutta.hpp"
#include "interpolation/interpolation_core.hpp"
#include <iostream>
#include <cmath>

// External global objects (defined in main executable)
extern SimulationConfig config;
extern SimulationData* sim;
extern RKData* rk;

void init()
{
    // Handle GPU configuration based on user preference and hardware availability
#if DMFE_WITH_CUDA
    if (config.gpu) {
        if (isCompatibleGPUInstalled()) {
            std::cout << "GPU acceleration enabled." << std::endl;
            config.gpu = true;
        } else {
            std::cout << "Warning: GPU acceleration requested but no compatible GPU found. Falling back to CPU." << std::endl;
            config.gpu = false;
        }
    } else {
        std::cout << "GPU acceleration disabled by user. Using CPU." << std::endl;
        config.gpu = false;
    }
#else
    std::cout << "Running in CPU-only mode." << std::endl;
    config.gpu = false;
#endif
    
    sim = new SimulationData();
    rk = new RKData();

    setupOutputDirectory();

    import(*sim, config.len, config.ord);

    // Generate filename based on parameters
    std::string filename = getFilename(config.resultsDir, config.p, config.p2, config.lambda, config.T0, config.Gamma, config.len, config.delta_t_min, config.delta_max, config.use_serk2, config.aggressive_sparsify, config.save_output);
    bool loaded = false;
    LoadedStateParams loaded_params;  // Declare the structure to hold loaded parameters

    // Try to load existing simulation data
    if (fileExists(filename) || fileExists(filename.substr(0, filename.find_last_of('.')) + ".bin")) {
        std::cout << "Found existing simulation file. Attempting to load..." << std::endl;
        loaded = loadSimulationState(filename, *sim, config.p, config.p2, config.lambda, config.T0, config.Gamma, config.len, config.delta_t_min, config.delta_max, config.use_serk2, config.aggressive_sparsify, loaded_params);
        if (loaded) {
            config.loaded = true;
            std::string dirPath = filename.substr(0, filename.find_last_of('/'));
            config.paramDir = dirPath;
        }
    }
    
    if (!loaded) {
        // Start new simulation with default values
        std::cout << "New simulation..." << std::endl;
        
        sim->h_t1grid.resize(1, 0.0);
        sim->h_delta_t_ratio.resize(1, 0.0);
        config.specRad = 4 * sqrt(DDflambda(1));

        config.delta_t = config.delta_t_min;
        config.loop = 0;
        config.delta = 1;
        config.delta_old = 0;

        sim->h_QKv.resize(config.len, 1.0);
        sim->h_QRv.resize(config.len, 1.0);
        sim->h_dQKv.resize(config.len, 0.0);
        sim->h_dQRv.resize(config.len, 0.0);
        sim->h_rvec.resize(1, config.Gamma + Dflambda(1) / config.T0);
        sim->h_drvec.resize(1, rstep());
    } else {
        // We successfully loaded data, use the loaded parameters
        config.delta = loaded_params.delta;
        config.delta_t = loaded_params.delta_t;
        config.loop = loaded_params.loop;
        config.specRad = 4 * sqrt(DDflambda(1));
        std::cout << "Loaded simulation state: delta=" << config.delta 
                  << ", delta_t=" << config.delta_t << ", loop=" << config.loop << std::endl;
    }

    // Initialize intermediate arrays needed for interpolation (always needed)
    sim->h_posB1xOld.resize(config.len, 1.0);
    sim->h_posB2xOld.resize(config.len * config.len, 0.0);

    sim->h_SigmaKA1int.resize(config.len * config.len, 0.0);
    sim->h_SigmaRA1int.resize(config.len * config.len, 0.0);
    sim->h_SigmaKB1int.resize(config.len * config.len, 0.0);
    sim->h_SigmaRB1int.resize(config.len * config.len, 0.0);
    sim->h_SigmaKA2int.resize(config.len * config.len, 0.0);
    sim->h_SigmaRA2int.resize(config.len * config.len, 0.0);
    sim->h_SigmaKB2int.resize(config.len * config.len, 0.0);
    sim->h_SigmaRB2int.resize(config.len * config.len, 0.0);

    sim->h_QKA1int.resize(config.len * config.len, 0.0);
    sim->h_QRA1int.resize(config.len * config.len, 0.0);
    sim->h_QKB1int.resize(config.len * config.len, 0.0);
    sim->h_QRB1int.resize(config.len * config.len, 0.0);
    sim->h_QKA2int.resize(config.len * config.len, 0.0);
    sim->h_QRA2int.resize(config.len * config.len, 0.0);
    sim->h_QKB2int.resize(config.len * config.len, 0.0);
    sim->h_QRB2int.resize(config.len * config.len, 0.0);

    sim->h_rInt.resize(config.len, 0.0);
    sim->h_drInt.resize(config.len, 0.0);

    if (config.gpu) {
#if DMFE_WITH_CUDA
        copyVectorsToGPU(*sim, config.len);
        copyParametersToDevice(config.p, config.p2, config.lambda);

        // (IndexVecLN3 optimizer setup no longer needed here; handled inside indexVecLN3GPU)
        // weightsB2y optimization setup now handled internally in indexMatAllGPU (if needed)

        // Initialize the appropriate GPU RK method to ensure required buffers are set
        if (config.delta_t < config.rmax[0] / config.specRad) {
            init_RK54GPU();   // RK54
        } else {
            init_SSPRK104GPU(); // SSPRK104
        }

        // Initial interpolation on GPU
        interpolateGPU();
#endif
    } else {
        // Choose CPU RK method consistent with previous version's behavior
        if (config.delta_t < config.rmax[0] / config.specRad) {
            rk->init = 1;  // RK54
        } else {
            rk->init = 2;  // SSPRK104
        }
        interpolate();
    }
}
