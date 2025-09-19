// Example builds (prefer CMake for portability)
// - NVCC + GCC/Clang host: keep -ffast-math in -Xcompiler
//   nvcc --extended-lambda --use_fast_math -gencode arch=compute_90,code=sm_90 -O3 \
//        -Xcompiler "-O3 -march=native -ffast-math" -o RG-Evo main.cu
// - NVCC + NVHPC (nvc++) host: DO NOT use -ffast-math; use -fast bundle instead
//   nvcc --extended-lambda --use_fast_math -gencode arch=compute_90,code=sm_90 -O3 \
//        -ccbin nvc++ -Xcompiler "-fast" -o RG-Evo main.cu
// - With HDF5 (static link mode): add -DUSE_HDF5 and host link libs as needed
//   ... -DUSE_HDF5 -lhdf5 -lhdf5_cpp

#include <iostream>
#include <chrono>
#include <thread>
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
    // Note: OpenMP initialization is handled in the parallel regions of other functions
    // The actual thread count will be determined by the OpenMP runtime in the computational functions
    
    std::cout << "DMFE Simulation starting..." << std::endl;
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

    // Initialize simulation
    init();

    // Run the main simulation
    return runSimulation();
}
