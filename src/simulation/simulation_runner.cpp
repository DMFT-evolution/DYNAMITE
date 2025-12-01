#include "simulation/simulation_runner.hpp"
#include "simulation/simulation_data.hpp"
#include "EOMs/rk_data.hpp"
#include "core/config.hpp"
#include "core/config_build.hpp"
#include "core/stream_pool.hpp"
#include "core/gpu_memory_utils.hpp"
#include "core/device_utils.cuh"
#include "io/io_utils.hpp"
#include "EOMs/time_steps.hpp"
#include "EOMs/runge_kutta.hpp"
#include "sparsify/sparsify_utils.hpp"
#include "interpolation/interpolation_core.hpp"
#include "simulation/simulation_control.hpp"
#include "convolution/convolution.hpp"
#include "core/vector_utils.hpp"
#include "math/math_sigma.hpp"
#include "core/host_utils.hpp"
#include "math/math_ops.hpp"
#include <iostream>
#include "core/console.hpp"
#include <fstream>
#include <iomanip>
#include <limits>
#include <chrono>
#include <algorithm>
#include <numeric>
#include "search/search_utils.hpp"
#include <unistd.h> // isatty
#include <sys/ioctl.h> // TIOCGWINSZ
#include <termios.h>
#include <cstdlib>  // getenv
#include <sstream>
#include <cmath>
#include <csignal>
#include <cstring>
#include "ui/simulation_text_ui.hpp"
#include "simulation/tail_fit.hpp"

// External global variables (defined in main.cpp)
extern SimulationConfig config;
extern SimulationData* sim;
extern RKData* rk;
extern size_t peak_memory_kb;
#if DMFE_WITH_CUDA
extern size_t peak_gpu_memory_mb;
#endif

// Global program_start_time definition moved to src/core/program_start_time.cpp to avoid
// nvcc host/device compilation issues and ensure a single, clear translation unit.

// Track last rollback iteration
int last_rollback_loop = -1000;

int runSimulation() {
#if DMFE_WITH_CUDA
    // Only create CUDA streams if we're actually using the GPU
    StreamPool* pool = config.gpu ? new StreamPool(20) : nullptr;
#endif

    dmfe::console::init();

    // Text UI
    dmfe::ui::SimulationTextUI ui;
    ui.install_signal_handlers();
    ui.start_clock();

    std::cout << dmfe::console::INFO() << "Starting simulation..." << std::endl;

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
            std::cerr << dmfe::console::ERR() << "Unable to open " << corrFilename << std::endl;
#if DMFE_WITH_CUDA
            delete pool;
#endif
            return 1;
        }
        corr << std::fixed << std::setprecision(14);
    } else {
    std::cout << dmfe::console::WARN() << "Output saving disabled. Correlation data will not be written to file." << std::endl;
    }

    std::cout << std::fixed << std::setprecision(14);

#if DMFE_WITH_CUDA
    double t = (config.gpu ? sim->d_t1grid.back() : sim->h_t1grid.back());
#else
    double t = sim->h_t1grid.back();
#endif

    // Baseline: last time from the loaded state. Used to decide whether to save outputs.
    const double baseline_t1_last = t;
    bool skip_save_notice_emitted = false;
    auto has_progress_beyond_loaded = [&]() -> bool {
#if DMFE_WITH_CUDA
        double cur = (config.gpu ? sim->d_t1grid.back() : sim->h_t1grid.back());
#else
        double cur = sim->h_t1grid.back();
#endif
        // Require strictly greater than baseline with a small numerical tolerance
        const double atol = 1e-12;
        const double rtol = 1e-12;
        const double tol = std::max(atol, rtol * std::fabs(baseline_t1_last));
        return (cur - baseline_t1_last) > tol;
    };

    // 2) Main loop
    // Status printing control
    const bool continuous_status = (!config.debug) && dmfe::console::stdout_is_tty();

    bool aborted_by_signal = false;
    while (t < config.tmax && config.loop < config.maxLoop && config.delta_t >= config.delta_t_min - std::numeric_limits<double>::epsilon() * std::max(std::abs(config.delta_t), std::abs(config.delta_t_min))) {
    if (ui.interrupted()) { aborted_by_signal = true; break; }
#if DMFE_WITH_CUDA
        t = (config.gpu ? sim->d_t1grid.back() : sim->h_t1grid.back()) + config.delta_t;
#else
        t = sim->h_t1grid.back() + config.delta_t;
#endif
        config.delta_old = config.delta;
#if DMFE_WITH_CUDA
    config.delta = (config.gpu ? updateGPU(pool) : update());
    // Optional tail fit/blend near theta->1 (toggle with --tail-fit)
    if (config.tail_fit_enabled) {
#if DMFE_WITH_CUDA
        if (config.gpu) { tailFitBlendGPU(); } else { tailFitBlendCPU(); }
#else
        tailFitBlendCPU();
#endif
    }
#else
        config.delta = update();
#endif
        config.loop++;

#if DMFE_WITH_CUDA
        if (config.loop % 1000 == 0) {
            updatePeakMemory();
        }

        if (config.loop % 100000 == 0) {
            // Update peak memory
            updatePeakMemory();
            
            // Auto mode: choose effective GPU sweep count based on current memory usage
            // Do not mutate config.sparsify_sweeps unless user provided explicit value.
            // We decide the effective sweep count below when computing max_sweeps.
#else
        if (config.loop % 100000 == 0) {
#endif
            
#if DMFE_WITH_CUDA
            size_t prev_size = config.gpu ? sim->d_t1grid.size() : sim->h_t1grid.size();
            int count = 0;
            // Determine effective sweeps:
            // - If user specified (>=0), use that.
            // - If auto (-1): CPU=1; GPU=1 or 2 depending on >50% GPU mem usage.
            int effective_sweeps = 1;
            if (config.sparsify_sweeps >= 0) {
                effective_sweeps = config.sparsify_sweeps;
            } else {
#if DMFE_WITH_CUDA
                if (config.gpu) {
                    size_t total_gpu_mb = getAvailableGPUMemory();
                    size_t used_gpu_mb = getGPUMemoryUsage();
                    effective_sweeps = (total_gpu_mb > 0 && used_gpu_mb > total_gpu_mb * 0.5) ? 2 : 1;
                } else {
                    effective_sweeps = 1;
                }
#else
                effective_sweeps = 1;
#endif
            }
            int max_sweeps = effective_sweeps;
#else
            size_t prev_size = sim->h_t1grid.size();
            int count = 0;
            int max_sweeps = 1;
#endif
            while (count < std::min(10, max_sweeps)) {
#if DMFE_WITH_CUDA
                if (config.gpu) {
                    sparsifyNscaleGPU(config.delta_max);
                } else {
#endif
                    sparsifyNscale(config.delta_max);
#if DMFE_WITH_CUDA
                }
                size_t new_size = config.gpu ? sim->d_t1grid.size() : sim->h_t1grid.size();
#else
                size_t new_size = sim->h_t1grid.size();
#endif
                if (new_size >= prev_size) break;
                prev_size = new_size;
                 count++;
            }

#if DMFE_WITH_CUDA
            if (config.gpu) {
                interpolateGPU();
            } else {
#endif
                interpolate();
#if DMFE_WITH_CUDA
            }
#endif

            if (config.delta < config.delta_max / 2 && config.loop - last_rollback_loop > 1000) {
                config.delta_t *= 0.5;
                if (config.gpu) {
#if DMFE_WITH_CUDA
                    if (rk->init == 1) {
                        init_SSPRK104GPU();
                    } else if (config.use_serk2) {
                        init_SERK2(2 * (rk->init - 1));
                    }
#endif
                } else {
                    rk->init = 2;
                } 
            }
            if (config.save_output) {
                // Use the same directory as correlation.txt for consistency
                std::string filename = paramDir + "/data.h5";
                
                // Save state before sparsifying (overwriting the same file),
                // but only if we actually progressed beyond the loaded t1grid.
                if (!config.loaded || has_progress_beyond_loaded()) {
                    saveSimulationState(filename, config.delta, config.delta_t); // Telemetry will print start/finish
                } else if (!skip_save_notice_emitted) {
                    std::cout << dmfe::console::INFO()
                              << "Skipping save: no progress beyond loaded state (t1grid_last="
                              << std::setprecision(14) << baseline_t1_last << ")" << std::endl;
                    skip_save_notice_emitted = true;
                }
            }
        }

        // primitive time-step adaptation
#if DMFE_WITH_CUDA
        if (config.delta < config.delta_max && config.loop > 5 &&
            (config.delta < 1.1 * config.delta_old || config.delta_old < config.delta_max/1000) &&
            config.rmax[rk->init-1] / config.specRad > config.delta_t && (config.gpu ? sim->d_delta_t_ratio.back() : sim->h_delta_t_ratio.back()) == 1.0)
#else
        if (config.delta < config.delta_max && config.loop > 5 &&
            (config.delta < 1.1 * config.delta_old || config.delta_old < config.delta_max/1000) &&
            config.rmax[rk->init-1] / config.specRad > config.delta_t && sim->h_delta_t_ratio.back() == 1.0)
#endif
        {
            config.delta_t *= 1.01;
        }
        else if (config.delta > 2 * config.delta_max && config.delta_t > config.delta_t_min) {
            config.delta_t *= 0.9;
        }
        if (rk->init == 2 && config.delta > config.delta_max && config.rmax[0] / config.specRad > config.delta_t) {
#if DMFE_WITH_CUDA
            if (config.gpu) {
                init_RK54GPU();
            } else {
#endif
                rk->init = 1;
#if DMFE_WITH_CUDA
            }
#endif
            config.delta_t *= 0.5;
        }
#if DMFE_WITH_CUDA
        if (config.delta > 2 * config.delta_max && config.gpu) {
            // Determine safe rollback window (need at least 2 past points; n < currentSize-1)
            size_t currentSize = sim->d_t1grid.size();
            int max_allowed = static_cast<int>(currentSize) - 2; // as per rollbackState guard
            int n = std::min(10, std::max(0, max_allowed));
            if (n > 0) {
                bool ok = rollbackState(n);
                if (!ok) {
                    std::cerr << dmfe::console::WARN() << "Rollback of " << n << " iterations failed; reducing step without rollback." << std::endl;
                } else {
                    last_rollback_loop = config.loop;
                    t = sim->d_t1grid.back() + config.delta_t;
                }
            } else {
                std::cerr << dmfe::console::WARN() << "Insufficient history to rollback; reducing step without rollback." << std::endl;
            }

            // Reduce step and re-initialize integrator
            config.delta_t = 0.5 * std::min(std::max(config.delta_t, config.delta_t_min), config.rmax[rk->init-1] / config.specRad);
            if (rk->init > 3) {
                init_SERK2(2 * (rk->init - 3));
            } else {
                init_SSPRK104GPU();
            }
        }

        size_t current_t1len = config.gpu ? sim->d_t1grid.size() : sim->h_t1grid.size();
        // When GPU is disabled, avoid reading from device memory to prevent implicit D->H copies
        double qk0 = config.gpu
            ? sim->d_QKv[(current_t1len - 1) * config.len + 0]
            : sim->h_QKv[(current_t1len - 1) * config.len + 0];
#else
        size_t current_t1len = sim->h_t1grid.size();
        double qk0 = sim->h_QKv[(current_t1len - 1) * config.len + 0];
#endif

        // Status: continuous multi-line TUI for TTY when debug is off; otherwise line-per-iteration
        std::string method = (rk->init == 1 ? "RK54" : rk->init == 2 ? "SSPRK104" : "SERK2(" + std::to_string(2 * (rk->init - 2)) + ")");
        if (continuous_status) {
            ui.update_status(t, config.tmax, config.loop, config.delta_t, method);
        } else if (config.debug) {
            ui.print_debug_line(t, config.loop, config.delta_t, config.delta, method, qk0, current_t1len);
        } else {
            ui.print_periodic_line(t, config.loop, config.delta_t, config.delta, method, qk0, current_t1len);
        }

    // record QK(t,0) to file (only if progressed beyond loaded baseline)
    if (config.save_output && (!config.loaded || has_progress_beyond_loaded())) {
#if DMFE_WITH_CUDA
            if (config.gpu) {
                double energy = energyGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, config.T0); 
                corr << t << "\t" << energy << "\t" << qk0 << "\n";
            } else {
#endif
                std::vector<double> temp(config.len, 0.0);
                SigmaK(getLastLenEntries(sim->h_QKv, config.len), temp);
                double energy = -(ConvA(temp, getLastLenEntries(sim->h_QRv, config.len), t)[0] + Dflambda(qk0)/config.T0); 
                corr << t << "\t" << energy << "\t" << qk0 << "\n";
#if DMFE_WITH_CUDA
            }
#endif
        }

        // Record debug runtime telemetry (host-side only, gated to avoid perf impact)
        if (config.debug) {
            double sim_time_sample;
#if DMFE_WITH_CUDA
            sim_time_sample = config.gpu ? sim->d_t1grid.back() : sim->h_t1grid.back();
#else
            sim_time_sample = sim->h_t1grid.back();
#endif
            double runtime_sample = getRuntimeSeconds();
            auto& times = sim->h_debug_step_times;
            auto& runtimes = sim->h_debug_step_runtimes;
            const double tol = 1e-12;
            if (times.size() > runtimes.size()) {
                // keep vectors in lockstep even if previous state was inconsistent
                runtimes.resize(times.size());
            }
            bool should_append = times.empty() || sim_time_sample > times.back() + tol || sim_time_sample < times.back() - tol;
            if (should_append) {
                times.push_back(sim_time_sample);
                runtimes.push_back(runtime_sample);
            } else if (!runtimes.empty()) {
                runtimes.back() = runtime_sample;
            } else {
                // Shouldn't happen, but keep vectors aligned
                runtimes.push_back(runtime_sample);
            }
        }
    }

    SimulationDataSnapshot* final_snapshot = nullptr;
    if (config.save_output) {
        // Use the established paramDir for consistency with correlation.txt
        std::string filename = paramDir + "/data.h5";
        if (!config.loaded || has_progress_beyond_loaded()) {
            SimulationDataSnapshot snapshot = saveSimulationState(filename, config.delta, config.delta_t);
            final_snapshot = new SimulationDataSnapshot(snapshot); // Store for final output
        } else if (!skip_save_notice_emitted) {
            std::cout << dmfe::console::INFO()
                      << "Skipping final save: no progress beyond loaded state (t1grid_last="
                      << std::setprecision(14) << baseline_t1_last << ")" << std::endl;
            skip_save_notice_emitted = true;
        }
    // Final save requested; telemetry and I/O steps will report progress
    }

    if (config.save_output) {
        if (!config.loaded || has_progress_beyond_loaded()) {
            saveCompressedData(paramDir);
            dmfe::console::end_status_line_if_needed(continuous_status);
            if (config.debug) {
                std::cout << dmfe::console::SAVE() << "Compressed outputs written under " << paramDir << std::endl;
            }
        } else if (config.debug) {
            dmfe::console::end_status_line_if_needed(continuous_status);
            std::cout << dmfe::console::INFO() << "Compressed outputs skipped: no progress beyond loaded state." << std::endl;
        }
    }
    if (config.async_export) {
        waitForAsyncSavesToComplete();
    }

#if DMFE_WITH_CUDA
    if (config.gpu && !config.save_output) {
        copyVectorsToCPU(*sim);
    } 
#endif

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
#if DMFE_WITH_CUDA
        if (config.gpu) {
            dmfe::console::end_status_line_if_needed(false);
            runPerformanceBenchmark();
        } else {
            dmfe::console::end_status_line_if_needed(false);
            runPerformanceBenchmarkCPU();
        }
#else
    dmfe::console::end_status_line_if_needed(false);
        runPerformanceBenchmarkCPU();
#endif
    } 
    
    // 3) Print the final results or an abort message via UI
    ui.print_final_results(aborted_by_signal,
                           output_delta_t, output_delta, output_loop,
                           output_t1grid_last, output_rvec_last, output_drvec_last,
                           output_QKv_last, output_QRv_last,
                           continuous_status);

    // 4) Close the file
    if (config.save_output) {
        corr.close();
    }

#if DMFE_WITH_CUDA
    if (config.gpu) {
        clearAllDeviceVectors(*sim);
    }
#endif

    delete sim;
    delete rk;
#if DMFE_WITH_CUDA
    delete pool;
#endif
    return aborted_by_signal ? 130 : 0;
}

#if DMFE_WITH_CUDA
void runPerformanceBenchmark() {
    StreamPool* pool = new StreamPool(20);
    
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; ++i) {
        interpolateGPU(); 
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> total = end - start;
    double avg_ms = total.count() / 1000;
    std::cout << dmfe::console::BENCH() << "Average wall time: " << avg_ms << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; ++i) {
        updateGPU(pool);
    }

    end = std::chrono::high_resolution_clock::now();

    total = end - start;
    avg_ms = total.count() / 100;
    std::cout << dmfe::console::BENCH() << "Average wall time: " << avg_ms << " ms" << std::endl;
    
    delete pool;
}
#endif

void runPerformanceBenchmarkCPU() {

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; ++i) {
        interpolate();
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> total = end - start;
    double avg_ms = total.count() / 1000;
    std::cout << dmfe::console::BENCH() << "Average CPU interpolation time: " << avg_ms << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; ++i) {
        update();
    }

    end = std::chrono::high_resolution_clock::now();

    total = end - start;
    avg_ms = total.count() / 100;
    std::cout << dmfe::console::BENCH() << "Average CPU update time: " << avg_ms << " ms" << std::endl;
}
