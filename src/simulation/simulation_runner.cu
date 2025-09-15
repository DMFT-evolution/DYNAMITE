#include "simulation_runner.hpp"
#include "simulation_data.hpp"
#include "rk_data.hpp"
#include "config.hpp"
#include "stream_pool.hpp"
#include "io_utils.hpp"
#include "gpu_memory_utils.hpp"
#include "time_steps.hpp"
#include "runge_kutta.hpp"
#include "sparsify_utils.hpp"
#include "interpolation_core.hpp"
#include "simulation_control.hpp"
#include "convolution.hpp"
#include "vector_utils.hpp"
#include "math_sigma.hpp"
#include "host_utils.hpp"
#include "math_ops.hpp"
#include "device_utils.cuh"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <chrono>

// External global variables (defined in main.cu)
extern SimulationConfig config;
extern SimulationData* sim;
extern RKData* rk;

// Global timing variable
std::chrono::high_resolution_clock::time_point program_start_time;

int runSimulation() {
    // Only create CUDA streams if we're actually using the GPU
    StreamPool* pool = config.gpu ? new StreamPool(20) : nullptr;

    std::cout << "Starting simulation..." << std::endl;

    // 1) Open the output file for correlation
    std::ofstream corr;
    std::string paramDir = getParameterDirPath(config.resultsDir, config.p, config.p2, config.lambda, config.T0, config.Gamma, config.len);
    if (config.save_output) {
        ensureDirectoryExists(paramDir);
        std::string corrFilename = paramDir + "/correlation.txt";
        corr.open(corrFilename);
        if (!corr) {
            std::cerr << "Error: Unable to open " << corrFilename << std::endl;
            delete pool;
            return 1;
        }
        corr << std::fixed << std::setprecision(14);
    } else {
        std::cout << "Output saving disabled. Correlation data will not be written to file." << std::endl;
    }

    std::cout << std::fixed << std::setprecision(14);

    double t = (config.gpu ? sim->d_t1grid.back() : sim->h_t1grid.back());

    // 2) Main loop
    while (t < config.tmax && config.loop < config.maxLoop && config.delta_t >= config.delta_t_min - std::numeric_limits<double>::epsilon() * std::max(std::abs(config.delta_t), std::abs(config.delta_t_min))) {
        t = (config.gpu ? sim->d_t1grid.back() : sim->h_t1grid.back()) + config.delta_t;
        config.delta_old = config.delta;
        config.delta = (config.gpu ? updateGPU(pool) : update());
        config.loop++;

        if (config.loop % 1000 == 0) {
            updatePeakMemory();
        }

        if (config.loop % 100000 == 0) {
            if (config.gpu) {
                sparsifyNscaleGPU(config.delta_max);
                interpolateGPU();
            } else {
                sparsifyNscale(config.delta_max);
                interpolate();
            }
            if (config.delta < config.delta_max / 2) {
                config.delta_t *= 0.5;
                if (config.gpu) {
                    if (rk->init == 1) {
                        init_SSPRK104GPU();
                    } else if (config.use_serk2) {
                        init_SERK2(2 * (rk->init - 1));
                    }
                } else {
                    rk->init = 2;
                } 
            }
            if (config.gpu && config.save_output) {
                // Get consistent filename based on physical parameters
                std::string filename = getFilename(config.resultsDir, config.p, config.p2, config.lambda, config.T0, config.Gamma, config.len, config.save_output);
                
                // Save state before sparsifying (overwriting the same file)
                saveSimulationState(filename, config.delta, config.delta_t);
            }
        }

        // primitive time-step adaptation
        if (config.delta < config.delta_max && config.loop > 5 &&
            (config.delta < 1.1 * config.delta_old || config.delta_old < config.delta_max/1000) &&
            config.rmax[rk->init-1] / config.specRad > config.delta_t && (config.gpu ? sim->d_delta_t_ratio.back() : sim->h_delta_t_ratio.back()) == 1.0)
        {
            config.delta_t *= 1.01;
        }
        else if (config.delta > 2 * config.delta_max && config.delta_t > config.delta_t_min) {
            config.delta_t *= 0.9;
        }
        if (rk->init == 2 && config.delta > config.delta_max && config.rmax[0] / config.specRad > config.delta_t) {
            if (config.gpu) {
                init_RK54GPU();
            } else {
                rk->init = 1;
            }
            config.delta_t *= 0.5;
        }
        if (config.delta > 2 * config.delta_max && config.gpu) {
            rollbackState(10);
            t = (config.gpu ? sim->d_t1grid.back() : sim->h_t1grid.back()) + config.delta_t;
            if (rk->init > 3) {
                init_SERK2(2 * (rk->init - 3));
            } else {
                init_SSPRK104GPU();
            }
        }

        size_t current_t1len = config.gpu ? sim->d_t1grid.size() : sim->h_t1grid.size();
        double qk0 = config.gpu ? sim->d_QKv[(current_t1len - 1) * config.len + 0] : 
                  sim->h_QKv[(current_t1len - 1) * config.len + 0];

        // display a video
        std::cout << " loop: " << config.loop
            << " time: " << t
            << " time step: " << config.delta_t
            << " delta: " << config.delta
            << " method: " << (rk->init == 1 ? "RK54" : rk->init == 2 ? "SSPRK104" : "SERK2(" + std::to_string(2 * (rk->init - 2)) + ")")
            << " QK: " << qk0
            << " length of t1grid: " << current_t1len
            << std::endl;

        // record QK(t,0) to file
        if (config.save_output) {
            if (config.gpu) {
                double energy = energyGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, config.T0); 
                corr << t << "\t" << energy << "\t" << qk0 << "\n";
            } else {
                std::vector<double> temp(config.len, 0.0);
                SigmaK(getLastLenEntries(sim->h_QKv, config.len), temp);
                double energy = -(ConvA(temp, getLastLenEntries(sim->h_QRv, config.len), t)[0] + Dflambda(qk0)/config.T0); 
                corr << t << "\t" << energy << "\t" << qk0 << "\n";
            }
        }
    }

    if (config.gpu && config.save_output) {
        std::string filename = getFilename(config.resultsDir, config.p, config.p2, config.lambda, config.T0, config.Gamma, config.len, config.save_output);
        saveSimulationState(filename, config.delta, config.delta_t);
    }

    if (config.save_output) {
        saveCompressedData(paramDir);
    }

    if (config.gpu && !config.save_output) {
        copyVectorsToCPU(*sim);
    } 

    if (config.gpu && config.debug) {
        runPerformanceBenchmark();
    } 

    // 3) Print the final results
    std::cout << "final delta_t: " << config.delta_t << std::endl;
    std::cout << "final delta:   " << config.delta << std::endl;
    std::cout << "final loop:    " << config.loop << std::endl;
    std::cout << "final t1grid:  " << sim->h_t1grid.back() << std::endl;
    std::cout << "final rvec:    " << sim->h_rvec.back() << std::endl;
    std::cout << "final drvec:   " << sim->h_drvec.back() << std::endl;
    std::cout << "final QKv:     " << sim->h_QKv[(sim->h_t1grid.size() - 1) * config.len] - 1 << std::endl;
    std::cout << "final QRv:     " << sim->h_QRv[(sim->h_t1grid.size() - 1) * config.len] - 1 << std::endl;
    std::cout << "Simulation finished." << std::endl;

    // 4) Close the file
    if (config.save_output) {
        corr.close();
    }

    if (config.gpu) {
        clearAllDeviceVectors(*sim);
    }

    delete sim;
    delete rk;
    delete pool;
    return 0;
}

void runPerformanceBenchmark() {
    StreamPool* pool = new StreamPool(20);
    
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; ++i) {
        interpolateGPU(); 
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> total = end - start;
    double avg_ms = total.count() / 1000;
    std::cout << "Average wall time: " << avg_ms << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; ++i) {
        updateGPU(pool);
    }

    end = std::chrono::high_resolution_clock::now();

    total = end - start;
    avg_ms = total.count() / 100;
    std::cout << "Average wall time: " << avg_ms << " ms" << std::endl;
    
    delete pool;
}
