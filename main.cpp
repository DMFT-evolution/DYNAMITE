// CPU-only build of DMFE simulation
#include <iostream>
#include <chrono>
#include <thread>
#include "include/simulation/simulation_data.hpp" // SimulationData definition
#include "include/EOMs/rk_data.hpp" // RKData definition
#include "include/version/version_info.hpp" // versioning system
#include "include/core/config.hpp" // simulation configuration struct
#include "include/core/config_build.hpp" // build-time configuration
#include "include/core/initialization.hpp" // main initialization function
#include "include/simulation/simulation_runner.hpp" // main simulation loop

using namespace std;

// Global configuration and data structures
SimulationConfig config;

SimulationData* sim = nullptr;
RKData* rk = nullptr;

int main(int argc, char **argv) {
    // Note: OpenMP initialization is handled in the parallel regions of other functions
    // The actual thread count will be determined by the OpenMP runtime in the computational functions
    
    std::cout << "DMFE Simulation starting (CPU-only build)..." << std::endl;
    std::cout << "Available CPU cores: " << std::thread::hardware_concurrency() << std::endl;
    
    // Record program start time
    program_start_time = std::chrono::high_resolution_clock::now();

    // Display version information
    std::cout << "RG-Evo: DMFE Simulation" << std::endl;
    std::cout << "=======================" << std::endl;
    std::cout << g_version_info.toString() << std::endl;
    std::cout << std::endl;

    // Parse command line arguments
    if (!parseCommandLineArguments(argc, argv)) {
        return 1;
    }

    // Force GPU mode to false in CPU-only build
    #if !DMFE_WITH_CUDA
    if (config.gpu) {
        std::cout << "Warning: CUDA support not compiled in. Forcing CPU mode." << std::endl;
        config.gpu = false;
    }
    #endif

    // Initialize simulation
    init();

    // Run the main simulation
    return runSimulation();
}
