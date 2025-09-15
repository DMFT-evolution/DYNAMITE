#include "initialization.hpp"
#include "config.hpp"
#include "simulation_data.hpp"
#include "rk_data.hpp"
#include "device_utils.cuh"
#include "io_utils.hpp"
#include "math_ops.hpp"
#include "time_steps.hpp"
#include "gpu_memory_utils.hpp"
#include "runge_kutta.hpp"
#include "interpolation_core.hpp"
#include <iostream>

// External global objects (defined in main executable)
extern SimulationConfig config;
extern SimulationData* sim;
extern RKData* rk;

void init()
{
    config.gpu = isCompatibleGPUInstalled();
    sim = new SimulationData();
    rk = new RKData();

    setupOutputDirectory();

    import(*sim, config.len, config.ord);

    // Generate filename based on parameters
    std::string filename = getFilename(config.resultsDir, config.p, config.p2, config.lambda, config.T0, config.Gamma, config.len, config.save_output);
    bool loaded = false;
    LoadedStateParams loaded_params;  // Declare the structure to hold loaded parameters

    // Try to load existing simulation data
    if (fileExists(filename) || fileExists(filename.substr(0, filename.find_last_of('.')) + ".bin")) {
        std::cout << "Found existing simulation file. Attempting to load..." << std::endl;
        loaded = loadSimulationState(filename, *sim, config.p, config.p2, config.lambda, config.T0, config.Gamma, config.len, config.delta_t_min, config.delta_max, loaded_params);
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
    copyVectorsToGPU(*sim, config.len);
        copyParametersToDevice(config.p, config.p2, config.lambda);

    // (IndexVecLN3 optimizer setup no longer needed here; handled inside indexVecLN3GPU)
    // weightsB2y optimization setup now handled internally in indexMatAllGPU (if needed)

        // Choose appropriate RK method based on delta_t and stability range
        if (config.delta_t < config.rmax[0] / config.specRad) {
            init_RK54GPU();
        } else if (config.delta_t < config.rmax[1] / config.specRad) {
            init_SSPRK104GPU();
        } else if (config.delta_t < config.rmax[2] / config.specRad) {
            init_SERK2(2);
        } else if (config.delta_t < config.rmax[3] / config.specRad) {
            init_SERK2(4);
        } else if (config.delta_t < config.rmax[4] / config.specRad) {
            init_SERK2(6);
        } else if (config.delta_t < config.rmax[5] / config.specRad) {
            init_SERK2(8);
        } else if (config.delta_t < config.rmax[6] / config.specRad) {
            init_SERK2(10);
        } else if (config.delta_t < config.rmax[7] / config.specRad) {
            init_SERK2(12);
        } else if (config.delta_t < config.rmax[8] / config.specRad) {
            init_SERK2(14);
        } else if (config.delta_t < config.rmax[9] / config.specRad) {
            init_SERK2(16);
        } else {
            init_SERK2(18);
        }
    } else {
        // Choose appropriate RK method for CPU version
        if (config.delta_t < config.rmax[0] / config.specRad) {
            rk->init = 1;  // RK54
        } else {
            rk->init = 2;  // SSPRK104
        }
    }

    // Run interpolation to initialize intermediate arrays
    if (config.gpu) {
        interpolateGPU();
    } else {
        interpolate();
    }
}
