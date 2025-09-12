// Definitions for global simulation parameters declared in globals.hpp
#include "globals.hpp"
#include "config.hpp"
#include "version_info.hpp"
#include "version_compat.hpp"
#include "io_utils.hpp"
#include <iostream>
#include <string>
#include <unistd.h>

// External declarations for global config object (defined in main executable)
extern SimulationConfig config;

int p = 3;
int p2 = 12;
double lambda = 0.3;
double TMCT = 0.805166;
double T0 = 1e50;
double Gamma = 0.0;
int maxLoop = 10000;
std::string resultsDir = "Results/";
std::string outputDir = "/nobackups/jlang/Results/";
bool debug = true;
bool save_output = true;

double tmax = 1e7;
double delta_t_min = 1e-5;
double delta_max = 1e-10;
double rmax[11] = {3, 13, 20, 80, 180, 320, 500, 720, 980, 1280, 1620};

double delta = 0.0;
double delta_old = 0.0;
int loop = 0;
double specRad = 0.0;
double delta_t = 0.0;
size_t len = 512;
int ord = 0;
bool gpu = false;

// Pointers are owned/defined in the main executable (main.cu)
extern SimulationData* sim;
extern RKData* rk;

// Parse command line arguments and update simulation parameters
bool parseCommandLineArguments(int argc, char **argv) {
    int opt;
    while ((opt = getopt(argc, argv, "p:q:l:T:G:m:t:d:e:L:D:s:hvc:")) != -1) {
        switch (opt) {
            case 'p':
                config.p = std::stoi(optarg);
                break;
            case 'q':
                config.p2 = std::stoi(optarg);
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
                config.maxLoop = std::stoi(optarg);
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
                config.len = std::stoi(optarg);
                break;
            case 'D':
                config.debug = (std::string(optarg) != "false");
                break;
            case 's':
                config.save_output = (std::string(optarg) != "false");
                break;
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
            case 'h':
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "Options:\n"
                          << "  -p INT     Set p parameter (default: " << config.p << ")\n"
                          << "  -q INT     Set p2 parameter (default: " << config.p2 << ")\n"
                          << "  -l FLOAT   Set lambda parameter (default: " << config.lambda << ")\n"
                          << "  -T FLOAT   Set T0 parameter (use 'inf' for infinity, default: " << (config.T0 >= 1e50 ? "inf" : std::to_string(config.T0)) << ")\n"
                          << "  -G FLOAT   Set Gamma parameter (default: " << config.Gamma << ")\n"
                          << "  -m INT     Set maximum number of loops (default: " << config.maxLoop << ")\n"
                          << "  -L INT     Set maximum number of loops (default: " << config.len << ")\n"
                          << "  -t FLOAT   Set maximum simulation time (default: " << config.tmax << ")\n"
                          << "  -d FLOAT   Set minimum time step (default: " << config.delta_t_min << ")\n"
                          << "  -e FLOAT   Set maximum error per step (default: " << config.delta_max << ")\n"
                          << "  -s BOOL    Enable output saving (correlation file, simulation state, compressed data)\n"
                          << "  -D BOOL    Set debug mode (default: " << (config.debug ? "true" : "false") << ")\n"
                          << "  -v         Display version information and exit\n"
                          << "  -c FILE    Check version compatibility of parameter file and exit\n"
                          << "  -h         Display this help message and exit\n";
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
              << "  save_output = " << (config.save_output ? "true" : "false") << "\n";
    return true;
}

