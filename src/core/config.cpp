// Definitions for global simulation parameters declared in globals.hpp
#include "core/globals.hpp"
#include "core/config.hpp"
#include "version/version_info.hpp"
#include "version/version_compat.hpp"
#include "io/io_utils.hpp"
#include <iostream>
#include "core/console.hpp"
#include <string>
#include <unistd.h>
#include <getopt.h>
#include <algorithm>   // [CPC-REV] std::max

// External declarations for global config object (defined in main executable)
extern SimulationConfig config;

// Pointers are owned/defined in the main executable (main.cpp)
// extern SimulationData* sim;
// extern RKData* rk;

// Parse command line arguments and update simulation parameters
bool parseCommandLineArguments(int argc, char **argv) {
    // Store the original command-line arguments
    config.command_line_args.clear();
    for (int i = 0; i < argc; ++i) {
        config.command_line_args.push_back(std::string(argv[i]));
    }
    
    int opt;
    int long_index = 0;
    static struct option long_opts[] = {
        {"lambda", required_argument, 0, 'l'},
        {"T0", required_argument, 0, 'T'},
        {"Gamma", required_argument, 0, 'G'},
        {"error", required_argument, 0, 'e'},
        {"check", required_argument, 0, 'c'},
        {"out-dir", required_argument, 0, 'o'},
    {"sparsify-sweeps", required_argument, 0, 'w'},
        {"gpu", required_argument, 0, 'g'},
        {"async-export", required_argument, 0, 'A'},
        {"help", no_argument, 0, 'h'},
        {"serk2", required_argument, 0, 'S'},
        {"allow-incompatible-versions", required_argument, 0, 'I'},
    {"log-response-interp", required_argument, 0, 'R'},
        {"tail-fit", required_argument, 0, 'F'},
        // [CPC-REV] new options (referee 2, point 2): selectable Runge-Kutta error
        // norm and maintenance-pass interval.  Long options only, so that every
        // existing command line keeps working unchanged.
        {"error-norm", required_argument, 0, 1001},
        {"sparsify-interval", required_argument, 0, 1003},
        // [CPC-REV] explicit, self-documenting aliases for the two temperatures.
        // '--Gamma'/-G is the BATH temperature T of Eqs. (7) and (13); '--T0'/-T is
        // the temperature of the equilibrium initial condition, entering only
        // through beta_0 = 1/T0.  The historical names are kept for compatibility.
        {"bath-temperature", required_argument, 0, 'G'},
        {"initial-temperature", required_argument, 0, 'T'},
        {0, 0, 0, 0}
    };
    // Include 'S:' to accept -S true|false
    while ((opt = getopt_long(argc, argv, "p:q:l:T:G:m:t:d:e:L:D:s:S:w:o:g:A:hvc:I:R:F:", long_opts, &long_index)) != -1) {
        switch (opt) {
            case 'p':
                config.p = static_cast<int>(std::stod(optarg));
                break;
            case 'q':
                config.p2 = static_cast<int>(std::stod(optarg));
                break;
            case 'l':
                config.lambda = std::stod(optarg);
                break;
            case 'T':
                if (std::string(optarg) == "inf") {
                    config.T0 = 1e50;
                } else {
                    config.T0 = std::stod(optarg);
                }
                break;
            case 'G':
                config.Gamma = std::stod(optarg);
                break;
            case 'm':
                config.maxLoop = static_cast<int>(std::stod(optarg));
                break;
            case 't':
                config.tmax = std::stod(optarg);
                break;
            case 'd':
                config.delta_t_min = std::stod(optarg);
                break;
            case 'e':
                config.delta_max = std::stod(optarg);
                break;
            case 'L':
                config.len = static_cast<int>(std::stod(optarg));
                break;
            case 'D':
                config.debug = (std::string(optarg) != "false");
                break;
            case 's':
                config.save_output = (std::string(optarg) != "false");
                break;
            case 'S':
                config.use_serk2 = (std::string(optarg) != "false");
                break;
            case 'w': {
                // sparsify sweeps: -1 = auto; 0 disables; >0 fixed count
                try {
                    config.sparsify_sweeps = std::stoi(optarg);
                } catch (...) {
                    std::cerr << dmfe::console::WARN() << "Invalid value for --sparsify-sweeps; using auto (-1)" << std::endl;
                    config.sparsify_sweeps = -1;
                }
                break;
            }
            case 'g':
                config.gpu = (std::string(optarg) != "false");
                break;
            case 'A':
                config.async_export = (std::string(optarg) != "false");
                break;
            case 'o': {
                std::string path = optarg ? std::string(optarg) : std::string();
                if (!path.empty() && path.back() != '/') path += '/';
                if (!path.empty()) {
                    // Override both to keep behavior consistent regardless of HOME location
                    config.resultsDir = path;
                    config.outputDir = path;
                }
                break;
            }
            case 'v':
                std::cout << dmfe::console::INFO() << g_version_info.toString() << std::endl;
                config.exit_success_after_parse = true;   // [CPC-REV] exit code 0
                return false;  // Exit after showing version
            case 'c':
                // Check version compatibility of a specific parameter file
                {
                    std::string paramFile = optarg;
                    if (!fileExists(paramFile)) {
                        std::cerr << dmfe::console::ERR() << "Parameter file " << paramFile << " not found." << std::endl;
                        return false;
                    }
                    
                    VersionAnalysis analysis = analyzeVersionCompatibility(paramFile);
                    std::cout << dmfe::console::INFO() << "Version compatibility check for " << paramFile << ": ";
                    
                    switch (analysis.level) {
                        case VersionCompatibility::IDENTICAL:
                            std::cout << dmfe::console::DONE() << "IDENTICAL" << std::endl;
                            break;
                        case VersionCompatibility::COMPATIBLE:
                            std::cout << dmfe::console::DONE() << "COMPATIBLE" << std::endl;
                            break;
                        case VersionCompatibility::WARNING:
                            std::cout << dmfe::console::WARN() << "WARNING" << std::endl;
                            break;
                        case VersionCompatibility::INCOMPATIBLE:
                            std::cout << dmfe::console::ERR() << "INCOMPATIBLE" << std::endl;
                            break;
                    }
                    
                    std::cout << dmfe::console::INFO() << "Current version: " << g_version_info.code_version << std::endl;
                    std::cout << dmfe::console::INFO() << "File version: " << analysis.file_version << std::endl;
                    
                    for (const auto& warning : analysis.warnings) {
                        std::cout << dmfe::console::WARN() << warning << std::endl;
                    }
                    for (const auto& error : analysis.errors) {
                        std::cout << dmfe::console::ERR() << error << std::endl;
                    }
                    
                    config.exit_success_after_parse = true;   // [CPC-REV] exit code 0
                    return false;  // Exit after checking
                }
            case 'I':
                config.allow_incompatible_versions = (std::string(optarg) != "false");
                break;
            case 'R':
                // Accept true/false
                config.log_response_interp = (std::string(optarg) != "false");
                break;
            case 'F':
                // Enable/disable tail-fit blending near theta->1
                config.tail_fit_enabled = (std::string(optarg) != "false");
                break;
            case 1001: {   // [CPC-REV] --error-norm
                const std::string v(optarg ? optarg : "sum");
                if      (v == "sum"  || v == "0") config.error_norm = 0;
                else if (v == "mean" || v == "1") config.error_norm = 1;
                else if (v == "rms"  || v == "2") config.error_norm = 2;
                else if (v == "max"  || v == "3") config.error_norm = 3;
                else {
                    std::cerr << dmfe::console::WARN() << "Unknown --error-norm '" << v
                              << "'; falling back to 'sum'." << std::endl;
                    config.error_norm = 0;
                }
                break;
            }
            case 1003:     // [CPC-REV] --sparsify-interval
                try {
                    config.sparsify_interval = std::max(1, std::stoi(optarg));
                } catch (...) {
                    std::cerr << dmfe::console::WARN() << "Invalid --sparsify-interval; using 100000" << std::endl;
                    config.sparsify_interval = 100000;
                }
                break;
            case 'h':
                std::cout << dmfe::console::INFO() << "Usage: " << argv[0] << " [options]\n"
                          << "Options:\n"
                          << "  -p INT                          Set p parameter (default: " << config.p << ")\n"
                          << "  -q INT                          Set p2 parameter (default: " << config.p2 << ")\n"
                          << "  -l, --lambda F                  Set lambda parameter (default: " << config.lambda << ")\n"
                          << "  -T, --T0, --initial-temperature F\n"
                          << "                                  Temperature of the equilibrium INITIAL condition;\n"
                          << "                                  enters the equations only through beta_0 = 1/T0.\n"
                          << "                                  Use 'inf' for a random (infinite-temperature) start\n"
                          << "                                  (default: " << (config.T0 >= 1e50 ? "inf" : std::to_string(config.T0)) << ")\n"
                          << "  -G, --Gamma, --bath-temperature F\n"
                          << "                                  BATH temperature T after the quench: it sets the\n"
                          << "                                  noise <xi xi> = 2 T delta(t-t') and the constant term\n"
                          << "                                  of the Lagrange multiplier mu(t) = T + ...\n"
                          << "                                  (default: " << config.Gamma << ")\n"
                          << "  -m INT                          Set maximum number of loops (default: " << config.maxLoop << ")\n"
                          << "  -L INT                          Set grid length N (default: " << config.len << ")\n"
                          << "  -t FLOAT                        Set maximum simulation time (default: " << config.tmax << ")\n"
                          << "  -d FLOAT                        Set minimum time step (default: " << config.delta_t_min << ")\n"
                          << "  -e, --error F                   Set maximum error per step (default: " << config.delta_max << ")\n"
                          << "  -o, --out-dir DIR               Set directory for all outputs (overrides defaults)\n"
                          << "  -s BOOL                         Enable output saving (correlation file, simulation state, compressed data)\n"
                          << "  -S, --serk2 BOOL                Use SERK2 method (default: true)\n"
                          << "  -w, --sparsify-sweeps INT      Number of sparsify sweeps per maintenance pass (-1=auto, 0=off) (default: " << config.sparsify_sweeps << ")\n"
                          << "  -g, --gpu BOOL                  Enable GPU acceleration (default: " << (config.gpu ? "true" : "false") << ")\n"
                          << "  -A, --async-export BOOL        Enable asynchronous data export (default: " << (config.async_export ? "true" : "false") << ")\n"
                          << "  -R, --log-response-interp BOOL Log-space interpolation for response QR (QK interpolated normally) (default: " << (config.log_response_interp ? "true" : "false") << ")\n"
                          << "  -F, --tail-fit BOOL            Enable QK tail-fit near theta->1 (default: " << (config.tail_fit_enabled ? "true" : "false") << ")\n"
                          << "      --error-norm STR            Norm of the Runge-Kutta error vector:\n"
                          << "                                  sum (default, extensive 1-norm) | mean | rms | max\n"
                          << "      --sparsify-interval INT     Accepted steps between maintenance passes\n"
                          << "                                  (sparsify + re-interpolate + checkpoint; default 100000)\n"
                          << "  -D BOOL                         Set debug mode (default: " << (config.debug ? "true" : "false") << ")\n"
                          << "  -I, --allow-incompatible-versions BOOL  Allow loading data saved with incompatible versions (default: " << (config.allow_incompatible_versions ? "true" : "false") << ")\n"
                          << "  -v                              Display version information and exit\n"
                          << "  -c, --check FILE                Check version compatibility of parameter file and exit\n"
                          << "  -h, --help                      Display this help message and exit\n";
                config.exit_success_after_parse = true;   // [CPC-REV] exit code 0
                return false;
            default:
                std::cerr << dmfe::console::ERR() << "Unknown option: " << static_cast<char>(optopt) << std::endl;
                std::cerr << dmfe::console::INFO() << "Run with -h for help" << std::endl;
                return false;
        }
    }

    // Print the selected parameters
    std::cout << dmfe::console::INFO() << "Running simulation with parameters:\n"
              << "  p = " << config.p << "\n"
              << "  p2 = " << config.p2 << "\n"
              << "  lambda = " << config.lambda << "\n"
              << "  T0 = " << (config.T0 >= 1e50 ? "inf" : std::to_string(config.T0)) << "\n"
              << "  Gamma = " << config.Gamma << "\n"
              << "  len = " << config.len << "\n"
              << "  maxLoop = " << config.maxLoop << "\n"
              << "  tmax = " << config.tmax << "\n"
              << "  delta_t_min = " << config.delta_t_min << "\n"
              << "  delta_max = " << config.delta_max << "\n"
              << "  debug = " << (config.debug ? "true" : "false") << "\n"
              << "  save_output = " << (config.save_output ? "true" : "false") << "\n"
              << "  use_serk2 = " << (config.use_serk2 ? "true" : "false") << "\n"
              << "  sparsify_sweeps = " << config.sparsify_sweeps << "\n"
              << "  gpu = " << (config.gpu ? "true" : "false") << "\n"
              << "  async_export = " << (config.async_export ? "true" : "false") << "\n"
              << "  allow_incompatible_versions = " << (config.allow_incompatible_versions ? "true" : "false") << "\n"
              << "  log_response_interp = " << (config.log_response_interp ? "true" : "false") << "\n"
              << "  tail_fit_enabled = " << (config.tail_fit_enabled ? "true" : "false") << "\n"
              << "  error_norm = " << errorNormName(config.error_norm) << "\n"
              << "  sparsify_interval = " << config.sparsify_interval << "\n"
              << "  out_dir = " << config.resultsDir << "\n";
    return true;
}

