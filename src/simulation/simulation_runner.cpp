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

// External global variables (defined in main.cpp)
extern SimulationConfig config;
extern SimulationData* sim;
extern RKData* rk;
extern size_t peak_memory_kb;
#if DMFE_WITH_CUDA
extern size_t peak_gpu_memory_mb;
#endif

// Global timing variable
std::chrono::high_resolution_clock::time_point program_start_time;

// Track last rollback iteration
int last_rollback_loop = -1000;

// Async-signal-safe terminal reset on abort (Ctrl-C etc.)
static volatile sig_atomic_t g_interrupted = 0;
static void dmfe_sig_handler(int /*sig*/){
    g_interrupted = 1;
    // Reset scroll region, ensure cursor visible, move near bottom, clear line, newline
    // Use async-signal-safe write() with fixed lengths; avoid iostreams here.
    const char seq1[] = "\033[r";       // reset scroll region
    const char seq2[] = "\033[?25h";    // show cursor (in case hidden elsewhere)
    const char seq3[] = "\033[999;1H";  // move cursor near bottom
    const char seq4[] = "\033[K";       // clear to end of line
    const char seq5[] = "\n";           // newline to release prompt area
    // write() is async-signal-safe
    (void)!write(STDOUT_FILENO, seq1, sizeof(seq1)-1);
    (void)!write(STDOUT_FILENO, seq2, sizeof(seq2)-1);
    (void)!write(STDOUT_FILENO, seq3, sizeof(seq3)-1);
    (void)!write(STDOUT_FILENO, seq4, sizeof(seq4)-1);
    (void)!write(STDOUT_FILENO, seq5, sizeof(seq5)-1);
}

int runSimulation() {
#if DMFE_WITH_CUDA
    // Only create CUDA streams if we're actually using the GPU
    StreamPool* pool = config.gpu ? new StreamPool(20) : nullptr;
#endif

    dmfe::console::init();

    // Install signal handlers to ensure terminal scroll region is reset on abort
    std::signal(SIGINT, dmfe_sig_handler);
    std::signal(SIGTERM, dmfe_sig_handler);
#ifdef SIGHUP
    std::signal(SIGHUP, dmfe_sig_handler);
#endif
    program_start_time = std::chrono::high_resolution_clock::now();

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

    // 2) Main loop
    // Status printing control
    const bool continuous_status = (!config.debug) && dmfe::console::stdout_is_tty();
    auto last_status_print = std::chrono::high_resolution_clock::now();
    // TUI placement: pin to bottom 4 lines using absolute positioning and a scroll region
    bool tui_scroll_region_set = false;
    int last_rows = -1;
    auto query_terminal_rows = []() -> int {
        struct winsize ws{};
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0) return ws.ws_row;
        const char* env = std::getenv("LINES");
        if (env) {
            int v = std::atoi(env);
            if (v > 0) return v;
        }
        return 24; // sensible default
    };
    auto set_scroll_region = [&](int rows){
        if (rows <= 6) return; // too small, skip
        // Reset region then set top..rows-4 as scrollable
        std::cout << "\033[r"; // reset to full screen
        std::cout << "\033[1;" << (rows - 4) << "r";
        tui_scroll_region_set = true;
        last_rows = rows;
    };
    auto ensure_scroll_region = [&](){
        int rows = query_terminal_rows();
        if (!tui_scroll_region_set || rows != last_rows) {
            set_scroll_region(rows);
        }
        return rows;
    };
    auto reset_scroll_region = [&](){
        if (tui_scroll_region_set) {
            std::cout << "\033[r" << std::flush; // reset
            tui_scroll_region_set = false;
            last_rows = -1;
        }
    };
    // Helper lambdas for pretty status output
    auto format_hms = [](double secs) {
        if (!std::isfinite(secs) || secs < 0) secs = 0;
        long s = static_cast<long>(secs + 0.5);
        long h = s / 3600; s %= 3600;
        long m = s / 60;   s %= 60;
        std::ostringstream os; os.fill('0');
        os << std::setw(2) << h << ":" << std::setw(2) << m << ":" << std::setw(2) << s;
        return os.str();
    };
    auto make_progress_bar = [](double frac, int width) {
        frac = std::max(0.0, std::min(1.0, frac));
        width = std::max(10, width);
        int filled = static_cast<int>(std::round(frac * width));
        if (filled > width) filled = width;
        if (filled < 0) filled = 0;
        std::string bar;
        bar.reserve(static_cast<size_t>(width) + 2);
        bar.push_back('[');
        // ASCII-only bar: '=' for filled, ' ' for remaining
        bar.append(static_cast<size_t>(filled), '=');
        bar.append(static_cast<size_t>(width - filled), ' ');
        bar.push_back(']');
        return bar;
    };
    auto make_mem_bar = [&](size_t used_kb, size_t total_kb, int width){
        if (total_kb == 0) return std::string();
        width = std::max(10, width);
        double frac = std::min(1.0, (double)used_kb / (double)total_kb);
        // Build plain ASCII bar with a minimum 1 filled slot for any non-zero usage
        int filled = static_cast<int>(std::floor(frac * width));
        if (frac > 0.0 && filled == 0) filled = 1; // ensure visibility at very low usage
        if (filled > width) filled = width;
        if (filled < 0) filled = 0;
        std::string bar;
        bar.reserve(static_cast<size_t>(width) + 2);
        bar.push_back('[');
        bar.append(static_cast<size_t>(filled), '=');
        bar.append(static_cast<size_t>(width - filled), ' ');
        bar.push_back(']');

        // Colorize bar based on thresholds if color output is enabled
        dmfe::console::init();
        const bool enable_color = dmfe::console::color_out();
        const char* color = dmfe::console::C_GREEN;
        if (frac >= 0.75) {
            color = dmfe::console::C_RED;
        } else if (frac >= 0.50) {
            color = dmfe::console::C_YELLOW;
        }
        if (enable_color) {
            return std::string(color) + bar + dmfe::console::C_RESET;
        }
        return bar;
    };

    bool aborted_by_signal = false;
    while (t < config.tmax && config.loop < config.maxLoop && config.delta_t >= config.delta_t_min - std::numeric_limits<double>::epsilon() * std::max(std::abs(config.delta_t), std::abs(config.delta_t_min))) {
        if (g_interrupted) { aborted_by_signal = true; break; }
#if DMFE_WITH_CUDA
        t = (config.gpu ? sim->d_t1grid.back() : sim->h_t1grid.back()) + config.delta_t;
#else
        t = sim->h_t1grid.back() + config.delta_t;
#endif
        config.delta_old = config.delta;
#if DMFE_WITH_CUDA
        config.delta = (config.gpu ? updateGPU(pool) : update());
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
            
            // Check memory usage and adjust sparsify sweeps
            if (config.gpu) {
                size_t available = getAvailableGPUMemory();
                if (peak_gpu_memory_mb > 0.5 * available) {
                    config.sparsify_sweeps = 2;
                } else {
                    config.sparsify_sweeps = 1;
                }
            }
#else
        if (config.loop % 100000 == 0) {
#endif
            
            if (config.aggressive_sparsify) {
#if DMFE_WITH_CUDA
                size_t prev_size = config.gpu ? sim->d_t1grid.size() : sim->h_t1grid.size();
                int count = 0;
                int max_sweeps = config.gpu ? config.sparsify_sweeps : 1; // For CPU, default to 1, but can adjust
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
            } else {
#if DMFE_WITH_CUDA
                if (config.gpu) {
                    for (int i = 0; i < config.sparsify_sweeps; ++i) {
                        sparsifyNscaleGPU(config.delta_max);
                    }
                } else {
#endif
                    sparsifyNscale(config.delta_max);
#if DMFE_WITH_CUDA
                }
#endif
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
                
                // Save state before sparsifying (overwriting the same file)
                saveSimulationState(filename, config.delta, config.delta_t); // Return value ignored for intermediate saves
                // Notify save (avoid clobbering status line)
                dmfe::console::end_status_line_if_needed(continuous_status);
                std::cout << dmfe::console::SAVE() << "Snapshot saved to " << filename << std::endl;
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
    if (continuous_status) {
            auto now = std::chrono::high_resolution_clock::now();
            // Throttle to ~5 Hz for smooth, efficient updates, but also refresh immediately if save telemetry changed
            extern bool consumeSaveTelemetryDirty();
            extern bool consumeStatusAnchorInvalidated();
            bool force_refresh = consumeSaveTelemetryDirty();
            bool reset_anchor = consumeStatusAnchorInvalidated();
            if (reset_anchor) {
                dmfe::console::end_status_line_if_needed(true);
            }
            if (force_refresh || std::chrono::duration_cast<std::chrono::milliseconds>(now - last_status_print).count() >= 200) {
                last_status_print = now;

                // Elapsed runtime and ETA
                double elapsed_s = std::chrono::duration<double>(now - program_start_time).count();
                double frac = (config.tmax > 0.0) ? std::max(0.0, std::min(1.0, t / config.tmax)) : 0.0;
                double eta_s = (frac > 1e-6) ? elapsed_s * (1.0 - frac) / frac : std::numeric_limits<double>::infinity();

                // Build progress bar (fixed width)
                const int bar_width = 28;
                std::string bar = make_progress_bar(frac, bar_width);

                // Method string
                std::string method = (rk->init == 1 ? "RK54" : rk->init == 2 ? "SSPRK104" : "SERK2(" + std::to_string(2 * (rk->init - 2)) + ")");
                // Memory info + bars
                extern size_t getCurrentMemoryUsage();
                extern size_t getTotalSystemMemoryKB();
                size_t cur_kb = 0;
                try { cur_kb = getCurrentMemoryUsage(); } catch (...) { cur_kb = 0; }
                size_t total_kb = getTotalSystemMemoryKB();
                std::ostringstream mem;
                if (cur_kb && total_kb) {
                    double frac_ram = std::min(1.0, (double)cur_kb / (double)total_kb);
                    const char* ram_color = dmfe::console::C_GREEN;
                    if (frac_ram >= 0.75) ram_color = dmfe::console::C_RED; else if (frac_ram >= 0.50) ram_color = dmfe::console::C_YELLOW;
                    if (dmfe::console::color_out()) mem << ram_color;
                    mem << "RAM " << (cur_kb/1024) << "/" << (total_kb/1024) << " MB ";
                    mem << make_mem_bar(cur_kb, total_kb, 20);
                    if (dmfe::console::color_out()) mem << dmfe::console::C_RESET;
                } else if (cur_kb) {
                    mem << "RAM " << (cur_kb/1024) << " MB";
                } else {
                    mem << "RAM peak " << (peak_memory_kb/1024) << " MB";
                }
#if DMFE_WITH_CUDA
                if (config.gpu) {
                    size_t total_gpu_mb = getAvailableGPUMemory(); // approximated total
                    size_t used_gpu_mb = getGPUMemoryUsage();
                    if (total_gpu_mb && used_gpu_mb) {
                        double frac_gpu = std::min(1.0, (double)used_gpu_mb / (double)total_gpu_mb);
                        const char* gpu_color = dmfe::console::C_GREEN;
                        if (frac_gpu >= 0.75) gpu_color = dmfe::console::C_RED; else if (frac_gpu >= 0.50) gpu_color = dmfe::console::C_YELLOW;
                        mem << " | "; if (dmfe::console::color_out()) mem << gpu_color;
                        mem << "GPU " << used_gpu_mb << "/" << total_gpu_mb << " MB ";
                        // Convert MB to KB for bar calculation
                        mem << make_mem_bar(used_gpu_mb*1024ULL, total_gpu_mb*1024ULL, 20);
                        if (dmfe::console::color_out()) mem << dmfe::console::C_RESET;
                        mem << " (peak " << peak_gpu_memory_mb << "MB)";
                    } else {
                        mem << " | GPU used " << used_gpu_mb << " MB (peak " << peak_gpu_memory_mb << "MB)";
                    }
                }
#endif

                // Save telemetry
                SaveTelemetry st = getSaveTelemetry();
                std::ostringstream save_line;
                if (!config.save_output) {
                    save_line << "(save disabled)";
                } else if (st.in_progress) {
                    double since = std::chrono::duration<double>(now - st.last_start_time).count();
                    save_line << "save in-progress -> " << st.target_file << " (" << format_hms(since) << ")";
                } else if (!st.last_completed_file.empty()) {
                    double ago = (st.last_end_time.time_since_epoch().count() > 0)
                        ? std::chrono::duration<double>(now - st.last_end_time).count() : 0.0;
                    save_line << "last save: " << st.last_completed_file << " (" << (ago>0?format_hms(ago):"now") << ")";
                } else {
                    save_line << "save: none yet";
                }

                // Compose multi-line block
                std::ostringstream line1, line2, line3, line4;
                line1.setf(std::ios::fixed);
                line1 << " time " << format_hms(elapsed_s)
                      << " | sim " << std::setprecision(6) << t << "/" << config.tmax
                      << " (" << std::setprecision(1) << (frac * 100.0) << "%) " << bar;
                line2.setf(std::ios::fixed);
                line2 << " dt " << std::setprecision(6) << config.delta_t
                      << " | meth " << method
                      << " | loop " << config.loop;
                if (std::isfinite(eta_s)) {
                    line2 << " | ETA " << format_hms(eta_s);
                }
                line3 << mem.str();
                line4 << save_line.str();

                // Ensure scroll region is configured and draw the 4-line TUI pinned to bottom
                int rows = ensure_scroll_region();
                if (rows > 6) {
                    int base = rows - 3; // first TUI line row
                    // Save current cursor, draw TUI using absolute positions, then restore cursor to bottom of scroll region
                    std::cout << "\0337\033[s"; // save cursor (DEC + CSI)
                    // Line 1
                    std::cout << "\033[" << base << ";1H" << dmfe::console::STAT() << line1.str() << "\033[K";
                    // Line 2
                    std::cout << "\033[" << (base + 1) << ";1H" << dmfe::console::STAT() << line2.str() << "\033[K";
                    // Line 3
                    std::cout << "\033[" << (base + 2) << ";1H" << dmfe::console::STAT() << line3.str() << "\033[K";
                    // Line 4
                    std::cout << "\033[" << (base + 3) << ";1H" << dmfe::console::STAT() << line4.str() << "\033[K";
                    // Restore cursor and move it to bottom of scroll region (rows-4, col 1) for logs
                    std::cout << "\0338\033[u"; // restore cursor
                    std::cout << "\033[" << (rows - 4) << ";1H" << std::flush;
                } else {
                    // Fallback: small terminal, print single-line status without control sequences
                    std::cout << dmfe::console::STAT() << line1.str() << " | " << line2.str() << "\n" << std::flush;
                }
            }
        } else if (config.debug) {
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << std::setprecision(14)
                << " loop: " << config.loop
                << " time: " << t
                << " time step: " << config.delta_t
                << " delta: " << config.delta
                << " method: "
                << (rk->init == 1 ? "RK54" : rk->init == 2 ? "SSPRK104" : "SERK2(" + std::to_string(2 * (rk->init - 2)) + ")")
                << " QK: " << qk0
                << " length of t1grid: " << current_t1len;
            std::cout << dmfe::console::STAT() << oss.str() << std::endl;
        } else {
            // Non-TTY and not debug: print periodically (e.g., once per second)
            auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_status_print).count() >= 1) {
                std::ostringstream oss;
                oss.setf(std::ios::fixed);
                oss << std::setprecision(14)
                    << " loop: " << config.loop
                    << " time: " << t
                    << " time step: " << config.delta_t
                    << " delta: " << config.delta
                    << " method: "
                    << (rk->init == 1 ? "RK54" : rk->init == 2 ? "SSPRK104" : "SERK2(" + std::to_string(2 * (rk->init - 2)) + ")")
                    << " QK: " << qk0
                    << " length of t1grid: " << current_t1len;
                std::cout << dmfe::console::STAT() << oss.str() << std::endl;
                last_status_print = now;
            }
        }

        // record QK(t,0) to file
        if (config.save_output) {
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
    }

    SimulationDataSnapshot* final_snapshot = nullptr;
    if (config.save_output) {
        // Use the established paramDir for consistency with correlation.txt
        std::string filename = paramDir + "/data.h5";
        SimulationDataSnapshot snapshot = saveSimulationState(filename, config.delta, config.delta_t);
        final_snapshot = new SimulationDataSnapshot(snapshot); // Store for final output
        dmfe::console::end_status_line_if_needed(continuous_status);
        std::cout << dmfe::console::SAVE() << "Final snapshot saved to " << filename << std::endl;
    }

    if (config.save_output) {
        saveCompressedData(paramDir);
        dmfe::console::end_status_line_if_needed(continuous_status);
        std::cout << dmfe::console::SAVE() << "Compressed outputs written under " << paramDir << std::endl;
    }

    // Wait for any async saves to complete before terminating (only if async mode is enabled)
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
    
    // 3) Print the final results or an abort message
    dmfe::console::end_status_line_if_needed(continuous_status);
    // Reset any scroll region and move cursor to last line before printing
    reset_scroll_region();
    std::cout << "\033[999;1H"; // move near bottom safely
    if (!aborted_by_signal) {
        std::cout << dmfe::console::RES() << "final delta_t: " << output_delta_t << std::endl;
        std::cout << dmfe::console::RES() << "final delta:   " << output_delta << std::endl;
        std::cout << dmfe::console::RES() << "final loop:    " << output_loop << std::endl;
        std::cout << dmfe::console::RES() << "final t1grid:  " << output_t1grid_last << std::endl;
        std::cout << dmfe::console::RES() << "final rvec:    " << output_rvec_last << std::endl;
        std::cout << dmfe::console::RES() << "final drvec:   " << output_drvec_last << std::endl;
        std::cout << dmfe::console::RES() << "final QKv:     " << output_QKv_last << std::endl;
        std::cout << dmfe::console::RES() << "final QRv:     " << output_QRv_last << std::endl;
        std::cout << dmfe::console::DONE() << "Simulation finished." << std::endl;
    } else {
        std::cout << dmfe::console::DONE() << "Simulation aborted by user (SIGINT)." << std::endl;
    }

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
