#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include "simulation/simulation_data.hpp"
#include "core/config.hpp"

// Struct to hold simulation data snapshot for background saving
struct SimulationDataSnapshot {
    std::vector<double> QKv, QRv, dQKv, dQRv, t1grid, rvec, drvec;
    std::vector<double> QKB1int, QRB1int, theta;
    double energy;
    double t_current;
    int current_len, current_loop;
    SimulationConfig config_snapshot;
    
    // Version and memory info for async saving
    std::string code_version, git_hash, git_branch, git_tag, build_date, build_time, compiler_version, cuda_version;
    bool git_dirty;
    size_t peak_memory_kb_snapshot, peak_gpu_memory_mb_snapshot;
    std::chrono::high_resolution_clock::time_point program_start_time_snapshot;
};

// Lightweight telemetry for async/sync save status (for TUI/status display)
struct SaveTelemetry {
    bool in_progress = false;           // true while background save thread is running
    std::string target_file;            // last target filename
    std::string last_completed_file;    // last file that finished saving
    std::chrono::high_resolution_clock::time_point last_start_time{}; // last save start
    std::chrono::high_resolution_clock::time_point last_end_time{};   // last save end
    // Progress info for TUI
    double progress = 0.0;              // 0.0 .. 1.0 approximate export progress
    double last_t_exported = 0.0;       // last time-step value exported in the last save
    std::string stage;                  // human-readable stage label (hdf5/params/histories/compressed)
};

// Query current save telemetry (thread-safe snapshot)
SaveTelemetry getSaveTelemetry();
// Internal: update/save telemetry helpers (defined in io_utils.cpp)
void _setSaveStart(const std::string& filename);
void _setSaveEnd(const std::string& filename);

// Update save progress (thread-safe)
void _setSaveProgress(double fraction, double last_t, const std::string& stage_label);
// Update only the last exported time value (thread-safe)
void _setSaveLastT(double last_t);

// Telemetry/UI coordination helpers
// Mark that save telemetry changed; runner should refresh status immediately.
void markSaveTelemetryDirty();
// Consume and clear the dirty flag; returns true if a refresh is requested.
bool consumeSaveTelemetryDirty();
// Mark that external prints occurred and TUI anchor should be reset.
void invalidateStatusAnchor();
// Consume and clear the anchor invalidation flag.
bool consumeStatusAnchorInvalidated();

// Structure to hold loaded simulation state parameters
struct LoadedStateParams {
    double delta, delta_t;
    int loop;
};

// Basic text file imports
std::vector<double> importVectorFromFile(const std::string& filename);
std::vector<size_t> importIntVectorFromFile(const std::string& filename);
void import(SimulationData& sim, size_t len_param, int& ord_ref); // bulk grid import

// Path helpers
void setupOutputDirectory();
std::string getParameterDirPath(const std::string& resultsDir_param, int p_param, int p2_param, 
                               double lambda_param, double T0_param, double Gamma_param, size_t len_param);
std::string findExistingParamDir(const std::string& resultsDir_param, int p_param, int p2_param,
                                double lambda_param, double T0_param, double Gamma_param, size_t len_param,
                                double delta_t_min_param, double delta_max_param, bool use_serk2_param);
void ensureDirectoryExists(const std::string& dir);
std::string getFilename(const std::string& resultsDir_param, int p_param, int p2_param, 
                       double lambda_param, double T0_param, double Gamma_param, size_t len_param, 
                       double delta_t_min_param, double delta_max_param, bool use_serk2_param,
                       bool save_output_param);

// File existence
bool fileExists(const std::string& filename);

// Loaders
bool loadSimulationStateBinary(const std::string& filename, SimulationData& sim,
                              int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param,
                              size_t len_param, double delta_t_min_param, double delta_max_param,
                              bool use_serk2_param,
                              LoadedStateParams& loaded_params);
#if defined(USE_HDF5)
bool loadSimulationStateHDF5(const std::string& filename, SimulationData& sim,
                            int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param,
                            size_t len_param, double delta_t_min_param, double delta_max_param,
                            bool use_serk2_param,
                            LoadedStateParams& loaded_params);
#endif
bool checkParametersMatch(const std::string& paramFilename, int p_param, int p2_param, double lambda_param, 
                         double T0_param, double Gamma_param, size_t len_param, double delta_t_min_param, double delta_max_param,
                         bool use_serk2_param);
bool loadSimulationState(const std::string& filename, SimulationData& sim,
                        int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param, 
                        size_t len_param, double delta_t_min_param, double delta_max_param,
                        bool use_serk2_param,
                        LoadedStateParams& loaded_params);

// History saving
void saveHistory(const std::string& filename, double delta, double delta_t, 
                 SimulationData& simulation, size_t len_param, double T0_param, bool gpu_param);

// Save functions
void saveParametersToFile(const std::string& dirPath, double delta, double delta_t);
// Async variant used by background HDF5 writer
void saveParametersToFileAsync(const std::string& dirPath, double delta, double delta_t, const SimulationDataSnapshot& snapshot);
void saveSimulationStateBinary(const std::string& filename, double delta, double delta_t);
// HDF5-based writers (compile-time or runtime optional)
#if defined(H5_RUNTIME_OPTIONAL) || defined(USE_HDF5)
void saveSimulationStateHDF5(const std::string& filename, double delta, double delta_t);
void saveSimulationStateHDF5Async(const std::string& filename, const SimulationDataSnapshot& snapshot);
#endif
SimulationDataSnapshot saveSimulationState(const std::string& filename, double delta, double delta_t);
void saveCompressedData(const std::string& dirPath);
void waitForAsyncSavesToComplete();

// GPU-side internal functions (defined in .cu files, called by .cpp)
SimulationDataSnapshot createDataSnapshot();
void saveHistoryAsync(const std::string& filename, double delta, double delta_t, const SimulationDataSnapshot& snapshot);
void saveCompressedDataAsync(const std::string& dirPath, const SimulationDataSnapshot& snapshot);

// Utility functions for save operations
std::string getCurrentTimestamp();
double getRuntimeSeconds();
std::string formatDuration(double total_seconds);
std::string getGPUInfo();
std::string formatMemory(size_t memory_kb);

// Helper functions for computation (used by saveHistory and other functions)
