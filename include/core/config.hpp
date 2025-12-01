#pragma once

#include <string>
#include <vector>

struct SimulationConfig {
    int p = 3;
    int p2 = 12;
    double lambda = 0.3;
    double TMCT = 0.805166;
    double T0 = 1e50;
    double Gamma = 0.0;
    int maxLoop = 10000;  // 1e4
    std::string resultsDir = "Results/";
    std::string outputDir = "/nobackups/jlang/Results/";
    bool debug = true;
    bool save_output = true;
    double tmax = 1e7;
    double delta_t_min = 1e-5;
    double delta_max = 1e-10;
    std::vector<double> rmax = {3, 13, 20, 80, 180, 320, 500, 720, 980, 1280, 1620};
    double delta = 0.0;
    double delta_old = 0.0;
    int loop = 0;
    double specRad = 0.0;
    double delta_t = 0.0;
    size_t len = 512;
    int ord = 0;
    bool gpu = true;
    bool use_serk2 = true;
    // Number of sparsification sweeps per maintenance pass.
    // -1 means "auto": CPU=1; GPU=1 or 2 depending on current GPU memory usage (>50% -> 2).
    // 0 disables sparsification.
    int sparsify_sweeps = -1;
    bool async_export = true;  // Enable asynchronous data export by default
    std::vector<std::string> command_line_args;  // Store original command-line arguments
    bool loaded = false;
    std::string paramDir;
    bool allow_incompatible_versions = false;  // Allow loading data saved with incompatible versions
    // Interpolate response QR and dQR in log space (safer when QR varies exponentially)
    bool log_response_interp = false;
    // Enable/disable QK tail-fit stabilization near theta->1
    bool tail_fit_enabled = false;
};

// Command line argument parsing
bool parseCommandLineArguments(int argc, char **argv);
