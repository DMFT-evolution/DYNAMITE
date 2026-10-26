#pragma once

#include <cstdlib>
#include <string>
#include <vector>

// Output root used when the executable lies inside the user's home directory (see
// setupOutputDirectory in src/io/io_utils.cpp).  By default this is the same as the
// general default, Results/ below the working directory.  A different directory can
// be compiled in with the CMake option DMFE_HOME_OUTPUT_DIR, e.g.
// "/nobackups/{USER}/Results/", where {USER} is replaced by the user name.
inline std::string default_output_dir() {
#ifdef DMFE_HOME_OUTPUT_DIR
    std::string dir = DMFE_HOME_OUTPUT_DIR;
    const std::string key = "{USER}";
    const std::string::size_type pos = dir.find(key);
    if (pos != std::string::npos) {
        const char *user = std::getenv("USER");
        if (!user || user[0] == '\0') user = std::getenv("LOGNAME");
        dir.replace(pos, key.size(), (user && user[0] != '\0') ? user : "unknown");
    }
    if (!dir.empty() && dir.back() != '/') dir += '/';
    return dir;
#else
    return "Results/";
#endif
}

struct SimulationConfig {
    int p = 3;
    int p2 = 12;
    double lambda = 0.3;
    double TMCT = 0.805166;
    double T0 = 1e50;
    double Gamma = 0.0;
    int maxLoop = 10000;  // 1e4
    std::string resultsDir = "Results/";
    std::string outputDir = default_output_dir();
    bool debug = true;
    bool save_output = true;
    double tmax = 1e7;
    double delta_t_min = 1e-5;
    double delta_max = 1e-10;
    // [CPC-REV] Norm applied to the 2L-component Runge-Kutta error vector
    // (dC_1..dC_L, dR_1..dR_L), see Eq. (25) of the manuscript and --error-norm:
    //   0 = extensive 1-norm  sum_i (|dC_i|+|dR_i|)              [default, legacy]
    //   1 = mean absolute     (1/2L) sum_i (|dC_i|+|dR_i|)
    //   2 = root mean square  sqrt( (1/2L) sum_i (dC_i^2+dR_i^2) )
    //   3 = maximum           max_i ( |dC_i|, |dR_i| )
    int error_norm = 0;
    // [CPC-REV] set by -h/-v/-c, which are informational and must exit with status 0.
    // Previously `./RG-Evo -v` returned 1, which makes it useless as a CI smoke test.
    bool exit_success_after_parse = false;
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
    // [CPC-REV] Number of accepted steps between two maintenance passes (sparsify +
    // re-interpolate + checkpoint).  Was hard-wired to 100000, which made the
    // convergence tests requested by referee 2 (point 2) impossible on short runs.
    int sparsify_interval = 100000;
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

// [CPC-REV] Name of the --error-norm setting, as accepted on the command line and
// written to params.txt.
inline const char* errorNormName(int norm) {
    switch (norm) {
        case 1:  return "mean";
        case 2:  return "rms";
        case 3:  return "max";
        default: return "sum";
    }
}

// Command line argument parsing
bool parseCommandLineArguments(int argc, char **argv);
