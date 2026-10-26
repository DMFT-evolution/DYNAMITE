// History files written at every checkpoint: energy.txt, rvec.txt, qk0.txt and, with
// -D true, step_metrics.txt.
//
// [CPC-REV] These writers used to live in io_output.cu, which is compiled only in
// CUDA builds, and every call was guarded by DMFE_WITH_CUDA, so CPU-only builds never
// wrote these files.  They are host code and are now compiled in every build; only the
// GPU evaluation of the energy history (energyHistoryGPU) remains in io_output.cu.
#include "io/io_utils.hpp"
#include "simulation/simulation_data.hpp"
#include "core/gpu_memory_utils.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#include "math/math_ops.hpp"
#include "math/math_sigma.hpp"
#include "convolution/convolution.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

extern SimulationConfig config;

void saveHistory(const std::string& filename, double delta, double delta_t, 
                 SimulationData& simulation, size_t len_param, double T0_param, bool gpu_param)
{
    // Ensure data is on CPU for saving
#if DMFE_WITH_CUDA
    if (gpu_param) {
        copyVectorsToCPU(simulation);
    }
#endif
    
    size_t t1len = simulation.host->t1grid.size();
    std::vector<double> energy_history(t1len);
    std::vector<double> qk0_history(t1len);
    
#if DMFE_WITH_CUDA
    if (gpu_param) {
        // Energy history on the GPU (energyHistoryGPU in io_output.cu)
        energyHistoryGPU(simulation, len_param, T0_param, energy_history);

        // Extract QK[0] values for each time step (parallelized on CPU side)
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)t1len; ++i) {
            qk0_history[i] = simulation.host->QKv[i * len_param];  // QKv[i * len] is QK[0] at time step i
        }
    } else
#endif
    {
        // CPU computation of energy history
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)t1len; ++i) {
            std::vector<double> temp(len_param, 0.0);
            std::vector<double> QKv_i(simulation.host->QKv.begin() + i * len_param,
                                      simulation.host->QKv.begin() + (i + 1) * len_param);
            std::vector<double> QRv_i(simulation.host->QRv.begin() + i * len_param,
                                      simulation.host->QRv.begin() + (i + 1) * len_param);
            SigmaK(QKv_i, temp);
            energy_history[i] = -(ConvA(temp, QRv_i, simulation.host->t1grid[i])[0] + flambda(QKv_i[0]) / T0_param);
            qk0_history[i] = QKv_i[0];  // QK[0] at time step i
        }
    }
    
    // Get directory path from filename
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    
    // Progress mapping for histories phase: [0.65 .. 0.80]
    const double p_start = 0.65;
    const double p_end   = 0.80;
    const double p_span  = (p_end - p_start);
    const double last_t1 = simulation.host->t1grid.empty() ? 0.0 : simulation.host->t1grid.back();
    auto update_hist_prog = [&](size_t done, size_t total){
        double frac = p_start + (total ? (p_span * (double)done / (double)total) : 0.0);
        if (frac > p_end) frac = p_end;
        if (frac < p_start) frac = p_start;
        _setSaveProgress(frac, last_t1, "histories");
    };
    const bool write_times = config.debug && !simulation.host->debug_step_runtimes.empty();
    // IMPORTANT: step_metrics is *not* sparsified/downsampled; we write all recorded debug telemetry.
    // Keep vector usage conservative: only consume indices that exist, and keep time/memory fallbacks.
    const size_t debug_sample_count = write_times ? simulation.host->debug_step_runtimes.size() : 0;
    const auto& memory_history = simulation.host->debug_step_memory;
    const bool memory_available = !memory_history.empty();
    size_t total_lines = simulation.host->t1grid.size() * 3 + debug_sample_count; // rvec + energy + qk0 (+times when debug)
    size_t done_lines = 0;
    update_hist_prog(done_lines, total_lines);

    // Save rvec history
    std::string rvecFilename = dirPath + "/rvec.txt";
    std::ofstream rvecFile(rvecFilename);
    if (rvecFile) {
        rvecFile << std::fixed << std::setprecision(16);
        rvecFile << "# Time\trvec\n";
        for (size_t i = 0; i < t1len; ++i) {
            rvecFile << simulation.host->t1grid[i] << "\t" << simulation.host->rvec[i] << "\n";
            if ((i & 0x3FF) == 0) { done_lines += 1024; update_hist_prog(done_lines, total_lines); }
        }
        rvecFile.close();
    } else {
    std::cerr << dmfe::console::ERR() << "Could not open file " << rvecFilename << std::endl;
    }
    
    // Save energy history
    std::string energyFilename = dirPath + "/energy.txt";
    std::ofstream energyFile(energyFilename);
    if (energyFile) {
        energyFile << std::fixed << std::setprecision(16);
        energyFile << "# Time\tEnergy\n";
        for (size_t i = 0; i < t1len; ++i) {
            energyFile << simulation.host->t1grid[i] << "\t" << energy_history[i] << "\n";
            if ((i & 0x3FF) == 0) { done_lines += 1024; update_hist_prog(done_lines, total_lines); }
        }
        energyFile.close();
    } else {
    std::cerr << dmfe::console::ERR() << "Could not open file " << energyFilename << std::endl;
    }
    
    // Save QK[0] history
    std::string qk0Filename = dirPath + "/qk0.txt";
    std::ofstream qk0File(qk0Filename);
    if (qk0File) {
        qk0File << std::fixed << std::setprecision(16);
        qk0File << "# Time\tQK[0]\n";
        for (size_t i = 0; i < t1len; ++i) {
            qk0File << simulation.host->t1grid[i] << "\t" << qk0_history[i] << "\n";
            if ((i & 0x3FF) == 0) { done_lines += 1024; update_hist_prog(done_lines, total_lines); }
        }
        qk0File.close();
    } else {
    std::cerr << dmfe::console::ERR() << "Could not open file " << qk0Filename << std::endl;
    }

    // Save wall-clock runtime history when debug telemetry is available
    if (write_times && debug_sample_count > 0) {
        std::string stepMetricsFilename = dirPath + "/step_metrics.txt";
        std::ofstream timesFile(stepMetricsFilename);
        if (timesFile) {
            timesFile << std::fixed << std::setprecision(16);
            std::string memory_header = gpu_param ? "GPUMemoryMB" : "RAMMemoryMB";
            timesFile << "# Time\tRuntimeSeconds\t" << memory_header << "\n";
            for (size_t i = 0; i < debug_sample_count; ++i) {
                // Use the debug time vector when present; otherwise fall back to closest t1grid entry.
                size_t safe_idx = t1len ? std::min(i, t1len - 1) : 0;
                double sim_time_value = (i < simulation.host->debug_step_times.size())
                    ? simulation.host->debug_step_times[i]
                    : (t1len ? simulation.host->t1grid[safe_idx] : 0.0);

                // Memory telemetry might be absent or shorter; use last-known value when available.
                double memory_value = (i < memory_history.size())
                    ? memory_history[i]
                    : (memory_available ? memory_history.back() : 0.0);

                timesFile << sim_time_value << "\t" << simulation.host->debug_step_runtimes[i]
                          << "\t" << memory_value << "\n";
                if ((i & 0x3FF) == 0) { done_lines += 1024; update_hist_prog(done_lines, total_lines); }
            }
            timesFile.close();
        } else {
            std::cerr << dmfe::console::ERR() << "Could not open file " << stepMetricsFilename << std::endl;
        }
    }
    
    // Consolidated summary line for history files
    if (config.debug) {
    std::cout << dmfe::console::SAVE() << "Saved histories (rvec, energy, qk0"
          << (write_times ? ", step_metrics" : "") << "; " << t1len 
                  << " time points) under " << dirPath << std::endl;
    }
    // Ensure we end histories phase at 0.80
    update_hist_prog(total_lines, total_lines);
}

// Async version of saveHistory that works with snapshot data (simplified, no GPU operations)
void saveHistoryAsync(const std::string& filename, double delta, double delta_t, const SimulationDataSnapshot& snapshot)
{
    // Get directory path from filename
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    
    size_t t1len = snapshot.t1grid.size();
    std::vector<double> energy_history(t1len);
    
    // CPU computation of energy history (same as synchronous version)
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)t1len; ++i) {
        std::vector<double> temp(snapshot.current_len, 0.0);
        std::vector<double> QKv_i(snapshot.QKv.begin() + i * snapshot.current_len,
                                  snapshot.QKv.begin() + (i + 1) * snapshot.current_len);
        std::vector<double> QRv_i(snapshot.QRv.begin() + i * snapshot.current_len,
                                  snapshot.QRv.begin() + (i + 1) * snapshot.current_len);
        SigmaK(QKv_i, temp);
        energy_history[i] = -(ConvA(temp, QRv_i, snapshot.t1grid[i])[0] + flambda(QKv_i[0]) / snapshot.config_snapshot.T0);
    }
    
    // Progress mapping for histories phase: [0.65 .. 0.80]
    const double p_start = 0.65;
    const double p_end   = 0.80;
    const double p_span  = (p_end - p_start);
    const double last_t1 = snapshot.t1grid.empty() ? 0.0 : snapshot.t1grid.back();
    auto update_hist_prog = [&](size_t done, size_t total){
        double frac = p_start + (total ? (p_span * (double)done / (double)total) : 0.0);
        if (frac > p_end) frac = p_end;
        if (frac < p_start) frac = p_start;
        _setSaveProgress(frac, last_t1, "histories");
    };
    const bool write_times = config.debug && !snapshot.debug_step_runtimes.empty();
    // IMPORTANT: step_metrics is *not* sparsified/downsampled; we write all recorded debug telemetry.
    const size_t debug_sample_count = write_times ? snapshot.debug_step_runtimes.size() : 0;
    const auto& memory_history = snapshot.debug_step_memory;
    const bool memory_available = !memory_history.empty();
    size_t total_lines = snapshot.t1grid.size() * 3 + debug_sample_count;
    size_t done_lines = 0;
    update_hist_prog(done_lines, total_lines);

    // Save rvec history
    std::string rvecFilename = dirPath + "/rvec.txt";
    std::ofstream rvecFile(rvecFilename);
    if (rvecFile) {
        rvecFile << std::fixed << std::setprecision(16);
        rvecFile << "# Time\trvec\n";
        for (size_t i = 0; i < t1len; ++i) {
            rvecFile << snapshot.t1grid[i] << "\t" << snapshot.rvec[i] << "\n";
            if ((i & 0x3FF) == 0) { done_lines += 1024; update_hist_prog(done_lines, total_lines); }
        }
        rvecFile.close();
    } else {
    std::cerr << dmfe::console::ERR() << "Could not open file " << rvecFilename << std::endl;
    }
    
    // Save energy history
    std::string energyFilename = dirPath + "/energy.txt";
    std::ofstream energyFile(energyFilename);
    if (energyFile) {
        energyFile << std::fixed << std::setprecision(16);
        energyFile << "# Time\tEnergy\n";
        for (size_t i = 0; i < t1len; ++i) {
            energyFile << snapshot.t1grid[i] << "\t" << energy_history[i] << "\n";
            if ((i & 0x3FF) == 0) { done_lines += 1024; update_hist_prog(done_lines, total_lines); }
        }
        energyFile.close();
    } else {
    std::cerr << dmfe::console::ERR() << "Could not open file " << energyFilename << std::endl;
    }
    
    // Save QK[0] history
    std::string qk0Filename = dirPath + "/qk0.txt";
    std::ofstream qk0File(qk0Filename);
    if (qk0File) {
        qk0File << std::fixed << std::setprecision(16);
        qk0File << "# Time\tQK[0]\n";
        for (size_t i = 0; i < t1len; ++i) {
            qk0File << snapshot.t1grid[i] << "\t" << snapshot.QKv[i * snapshot.current_len] << "\n";
            if ((i & 0x3FF) == 0) { done_lines += 1024; update_hist_prog(done_lines, total_lines); }
        }
        qk0File.close();
    } else {
    std::cerr << dmfe::console::ERR() << "Could not open file " << qk0Filename << std::endl;
    }

    if (write_times && debug_sample_count > 0) {
        std::string stepMetricsFilename = dirPath + "/step_metrics.txt";
        std::ofstream timesFile(stepMetricsFilename);
        if (timesFile) {
            timesFile << std::fixed << std::setprecision(16);
            std::string memory_header = snapshot.config_snapshot.gpu ? "GPUMemoryMB" : "RAMMemoryMB";
            timesFile << "# Time\tRuntimeSeconds\t" << memory_header << "\n";
            for (size_t i = 0; i < debug_sample_count; ++i) {
                size_t safe_idx = t1len ? std::min(i, t1len - 1) : 0;
                double sim_time_value = (i < snapshot.debug_step_times.size())
                    ? snapshot.debug_step_times[i]
                    : (t1len ? snapshot.t1grid[safe_idx] : 0.0);
                double memory_value = (i < memory_history.size())
                    ? memory_history[i]
                    : (memory_available ? memory_history.back() : 0.0);
                timesFile << sim_time_value << "\t" << snapshot.debug_step_runtimes[i]
                          << "\t" << memory_value << "\n";
                if ((i & 0x3FF) == 0) { done_lines += 1024; update_hist_prog(done_lines, total_lines); }
            }
            timesFile.close();
        } else {
            std::cerr << dmfe::console::ERR() << "Could not open file " << stepMetricsFilename << std::endl;
        }
    }
    
    // Consolidated summary line for history files (async)
    if (config.debug) {
    std::cout << dmfe::console::SAVE() << "Saved histories (rvec, energy, qk0"
        << (write_times ? ", step_metrics" : "") << "; " << t1len 
          << " time points) under " << dirPath << " (async)" << std::endl;
    }
    update_hist_prog(total_lines, total_lines);
}
