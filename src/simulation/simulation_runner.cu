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
#include <algorithm>
#include <numeric>
#include "search_utils.hpp"

// External global variables (defined in main.cu)
extern SimulationConfig config;
extern SimulationData* sim;
extern RKData* rk;

// Global timing variable
std::chrono::high_resolution_clock::time_point program_start_time;

// Track last rollback iteration
int last_rollback_loop = -1000;

int runSimulation() {
    // Only create CUDA streams if we're actually using the GPU
    StreamPool* pool = config.gpu ? new StreamPool(20) : nullptr;

    std::cout << "Starting simulation..." << std::endl;

    // 1) Open the output file for correlation
    std::ofstream corr;
    std::string paramDir;
    if (config.loaded) {
        paramDir = config.paramDir;
    } else {
        paramDir = getParameterDirPath(config.resultsDir, config.p, config.p2, config.lambda, config.T0, config.Gamma, config.len);
        if (config.save_output) {
            ensureDirectoryExists(paramDir);
        }
    }
    if (config.save_output) {
        std::string corrFilename = paramDir + "/correlation.txt";
        corr.open(corrFilename, std::ios::app);
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
            // Update peak memory
            updatePeakMemory();
            
            // Check memory usage and adjust sparsify sweeps
            if (config.gpu) {
                size_t available = getAvailableGPUMemory();
                if (peak_gpu_memory_mb > 0.5 * available) {
                    config.sparsify_sweeps = 2;
                } else {
                    config.sparsify_sweeps = 1;
                }
            }
            
            if (config.aggressive_sparsify) {
                size_t prev_size = config.gpu ? sim->d_t1grid.size() : sim->h_t1grid.size();
                int count = 0;
                int max_sweeps = config.gpu ? config.sparsify_sweeps : 1; // For CPU, default to 1, but can adjust
                while (count < std::min(10, max_sweeps)) {
                    if (config.gpu) {
                        sparsifyNscaleGPU(config.delta_max);
                    } else {
                        sparsifyNscale(config.delta_max);
                    }
                    size_t new_size = config.gpu ? sim->d_t1grid.size() : sim->h_t1grid.size();
                    if (new_size >= prev_size) break;
                    prev_size = new_size;
                    count++;
                }
            } else {
                if (config.gpu) {
                    for (int i = 0; i < config.sparsify_sweeps; ++i) {
                        sparsifyNscaleGPU(config.delta_max);
                    }
                } else {
                    sparsifyNscale(config.delta_max);
                }
            }

            if (config.gpu) {
                interpolateGPU();
            } else {
                interpolate();
            }

            if (config.delta < config.delta_max / 2 && config.loop - last_rollback_loop > 1000) {
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
            if (config.save_output) {
                // Use the same directory as correlation.txt for consistency
                std::string filename = paramDir + "/data.h5";
                
                // Save state before sparsifying (overwriting the same file)
                saveSimulationState(filename, config.delta, config.delta_t); // Return value ignored for intermediate saves
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
            last_rollback_loop = config.loop;
            t = (config.gpu ? sim->d_t1grid.back() : sim->h_t1grid.back()) + config.delta_t;
            config.delta_t = 0.5 * std::min(std::max(config.delta_t, config.delta_t_min), config.rmax[rk->init-1] / config.specRad);
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

    SimulationDataSnapshot* final_snapshot = nullptr;
    if (config.save_output) {
        // Use the established paramDir for consistency with correlation.txt
        std::string filename = paramDir + "/data.h5";
        SimulationDataSnapshot snapshot = saveSimulationState(filename, config.delta, config.delta_t);
        final_snapshot = new SimulationDataSnapshot(snapshot); // Store for final output
    }

    if (config.save_output) {
        saveCompressedData(paramDir);
    }

    // Wait for any async saves to complete before terminating (only if async mode is enabled)
    if (config.async_export) {
        waitForAsyncSavesToComplete();
    }

    if (config.gpu && !config.save_output) {
        copyVectorsToCPU(*sim);
    } 

    double output_delta_t = config.delta_t;
    double output_delta = config.delta;
    int output_loop = config.loop;
    double output_t1grid_last = sim->h_t1grid.back();
    double output_rvec_last = sim->h_rvec.back();
    double output_drvec_last = sim->h_drvec.back();
    double output_QKv_last = sim->h_QKv[(sim->h_t1grid.size() - 1) * config.len];
    double output_QRv_last = sim->h_QRv[(sim->h_t1grid.size() - 1) * config.len];

    // Use snapshot data for final output if async saving was used
    if (final_snapshot != nullptr) {
        output_t1grid_last = final_snapshot->t1grid.back();
        output_rvec_last = final_snapshot->rvec.back();
        output_drvec_last = final_snapshot->drvec.back();
        output_QKv_last = final_snapshot->QKv[(final_snapshot->t1grid.size() - 1) * final_snapshot->current_len];
        output_QRv_last = final_snapshot->QRv[(final_snapshot->t1grid.size() - 1) * final_snapshot->current_len];
        delete final_snapshot; // Clean up
    }

    if (config.debug) {
        if (config.gpu) {
            runPerformanceBenchmark();
        } else {
            runPerformanceBenchmarkCPU();
        }
    } 

    // 3) Print the final results
    std::cout << "final delta_t: " << output_delta_t << std::endl;
    std::cout << "final delta:   " << output_delta << std::endl;
    std::cout << "final loop:    " << output_loop << std::endl;
    std::cout << "final t1grid:  " << output_t1grid_last << std::endl;
    std::cout << "final rvec:    " << output_rvec_last << std::endl;
    std::cout << "final drvec:   " << output_drvec_last << std::endl;
    std::cout << "final QKv:     " << output_QKv_last << std::endl;
    std::cout << "final QRv:     " << output_QRv_last << std::endl;
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

void runPerformanceBenchmarkCPU() {

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; ++i) {
        interpolate();
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> total = end - start;
    double avg_ms = total.count() / 1000;
    std::cout << "Average CPU interpolation time: " << avg_ms << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; ++i) {
        update();
    }

    end = std::chrono::high_resolution_clock::now();

    total = end - start;
    avg_ms = total.count() / 100;
    std::cout << "Average CPU update time: " << avg_ms << " ms" << std::endl;
}
