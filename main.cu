//Compile on the cluster with nvcc --extended-lambda --use_fast_math -gencode arch=compute_90,code=sm_90 -O3 -Xcompiler "-O3 -march=native -ffast-math" -DUSE_HDF5 -o RG-Evo main.cu -lhdf5 -lhdf5_cpp
// Compile locally with nvcc -ccbin clang++ --extended-lambda --use_fast_math -O3 -Xcompiler "-O3 -march=native -ffast-math" -o RG-Evo main.cu

#include <iostream>
#include <chrono>
#include "include/simulation_data.hpp" // SimulationData definition
#include "include/rk_data.hpp" // RKData definition
#include "include/version_info.hpp" // versioning system
#include "include/config.hpp" // simulation configuration struct
#include "include/initialization.hpp" // main initialization function
#include "include/simulation_runner.hpp" // main simulation loop

using namespace std;

// Global configuration and data structures
SimulationConfig config;

#include "include/device_constants.hpp" // centralized device constant declarations

SimulationData* sim = nullptr;
RKData* rk = nullptr;

int main(int argc, char **argv) {
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

    // Initialize simulation
    init();

    // Run the main simulation
    return runSimulation();
}
