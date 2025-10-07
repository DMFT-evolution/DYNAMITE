// Definitions for global simulation parameters declared in globals.hpp
#include "core/globals.hpp"
#include "core/config.hpp"
#include "version/version_info.hpp"
#include "version/version_compat.hpp"
#include "io/io_utils.hpp"
#include <iostream>
#include <string>
#include <unistd.h>
#include <getopt.h>

// External declarations for global config object (defined in main executable)
extern SimulationConfig config;

// Pointers are owned/defined in the main executable (main.cpp)
extern SimulationData* sim;
extern RKData* rk;

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
        {"aggressive-sparsify", required_argument, 0, 'a'},
        {"gpu", required_argument, 0, 'g'},
        {"async-export", required_argument, 0, 'A'},
        {"help", no_argument, 0, 'h'},
        {"serk2", required_argument, 0, 'S'},
        {"allow-incompatible-versions", required_argument, 0, 'I'},
        {0, 0, 0, 0}
    };
    // Include 'S:' to accept -S true|false
    while ((opt = getopt_long(argc, argv, "p:q:l:T:G:m:t:d:e:L:D:s:S:a:o:g:A:hvc:I:", long_opts, &long_index)) != -1) {
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
            case 'a':
                config.aggressive_sparsify = (std::string(optarg) != "false");
                break;
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
                std::cout << g_version_info.toString() << std::endl;
                return false;  // Exit after showing version
            case 'c':
                // Check version compatibility of a specific parameter file
                {
                    std::string paramFile = optarg;
                    if (!fileExists(paramFile)) {
                        std::cerr << "Error: Parameter file " << paramFile << " not found." << std::endl;
                        return false;
                    }
                    
                    VersionAnalysis analysis = analyzeVersionCompatibility(paramFile);
                    std::cout << "Version compatibility check for " << paramFile << ": ";
                    
                    switch (analysis.level) {
                        case VersionCompatibility::IDENTICAL:
                            std::cout << "IDENTICAL" << std::endl;
                            break;
                        case VersionCompatibility::COMPATIBLE:
                            std::cout << "COMPATIBLE" << std::endl;
                            break;
                        case VersionCompatibility::WARNING:
                            std::cout << "WARNING" << std::endl;
                            break;
                        case VersionCompatibility::INCOMPATIBLE:
                            std::cout << "INCOMPATIBLE" << std::endl;
                            break;
                    }
                    
                    std::cout << "Current version: " << g_version_info.code_version << std::endl;
                    std::cout << "File version: " << analysis.file_version << std::endl;
                    
                    for (const auto& warning : analysis.warnings) {
                        std::cout << "Warning: " << warning << std::endl;
                    }
                    for (const auto& error : analysis.errors) {
                        std::cout << "Error: " << error << std::endl;
                    }
                    
                    return false;  // Exit after checking
                }
            case 'I':
                config.allow_incompatible_versions = (std::string(optarg) != "false");
                break;
            case 'h':
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "Options:\n"
                          << "  -p INT                          Set p parameter (default: " << config.p << ")\n"
                          << "  -q INT                          Set p2 parameter (default: " << config.p2 << ")\n"
                          << "  -l, --lambda F                  Set lambda parameter (default: " << config.lambda << ")\n"
                          << "  -T, --T0 F                      Set T0 parameter (use 'inf' for infinity, default: " << (config.T0 >= 1e50 ? "inf" : std::to_string(config.T0)) << ")\n"
                          << "  -G, --Gamma F                   Set Gamma parameter (default: " << config.Gamma << ")\n"
                          << "  -m INT                          Set maximum number of loops (default: " << config.maxLoop << ")\n"
                          << "  -L INT                          Set grid length N (default: " << config.len << ")\n"
                          << "  -t FLOAT                        Set maximum simulation time (default: " << config.tmax << ")\n"
                          << "  -d FLOAT                        Set minimum time step (default: " << config.delta_t_min << ")\n"
                          << "  -e, --error F                   Set maximum error per step (default: " << config.delta_max << ")\n"
                          << "  -o, --out-dir DIR               Set directory for all outputs (overrides defaults)\n"
                          << "  -s BOOL                         Enable output saving (correlation file, simulation state, compressed data)\n"
                          << "  -S, --serk2 BOOL                Use SERK2 method (default: true)\n"
                          << "  -a, --aggressive-sparsify BOOL  Enable aggressive sparsification (default: " << (config.aggressive_sparsify ? "true" : "false") << ")\n"
                          << "  -g, --gpu BOOL                 Enable GPU acceleration (default: " << (config.gpu ? "true" : "false") << ")\n"
                          << "  -A, --async-export BOOL        Enable asynchronous data export (default: " << (config.async_export ? "true" : "false") << ")\n"
                          << "  -D BOOL                         Set debug mode (default: " << (config.debug ? "true" : "false") << ")\n"
                          << "  -I, --allow-incompatible-versions BOOL  Allow loading data saved with incompatible versions (default: " << (config.allow_incompatible_versions ? "true" : "false") << ")\n"
                          << "  -v                              Display version information and exit\n"
                          << "  -c, --check FILE                Check version compatibility of parameter file and exit\n"
                          << "  -h, --help                      Display this help message and exit\n";
                return false;
            default:
                std::cerr << "Unknown option: " << static_cast<char>(optopt) << std::endl;
                std::cerr << "Run with -h for help" << std::endl;
                return false;
        }
    }

    // Print the selected parameters
    std::cout << "Running simulation with parameters:\n"
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
              << "  aggressive_sparsify = " << (config.aggressive_sparsify ? "true" : "false") << "\n"
              << "  gpu = " << (config.gpu ? "true" : "false") << "\n"
              << "  async_export = " << (config.async_export ? "true" : "false") << "\n"
              << "  allow_incompatible_versions = " << (config.allow_incompatible_versions ? "true" : "false") << "\n"
              << "  out_dir = " << config.resultsDir << "\n";
    return true;
}

