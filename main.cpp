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
#include "core/console.hpp"
#include <chrono>
#include <thread>
#include "include/simulation/simulation_data.hpp" // SimulationData definition
#include "include/EOMs/rk_data.hpp" // RKData definition
#include "include/version/version_info.hpp" // versioning system
#include "include/core/config.hpp" // simulation configuration struct
#include "include/core/config_build.hpp" // build-time configuration
#include "include/core/initialization.hpp" // main initialization function
#include "include/simulation/simulation_runner.hpp" // main simulation loop
#include "include/grid/theta_grid.hpp" // grid generation tools
#include "include/grid/phi_grid.hpp"   // phi grid generation and IO/validation
#include "include/grid/integration.hpp" // integration weights
#include "include/grid/pos_grid.hpp" // position grids from interpolation
#include "include/grid/grid_io.hpp"   // unified grid writer
#include "include/grid/grid_cli.hpp"  // grid subcommand handler

using namespace std;

// Global configuration and data structures
SimulationConfig config;

#include "include/core/device_constants.hpp" // centralized device constant declarations

SimulationData* sim = nullptr;
RKData* rk = nullptr;

int main(int argc, char **argv) {
    // Note: OpenMP initialization is handled in the parallel regions of other functions
    // The actual thread count will be determined by the OpenMP runtime in the computational functions
    
    // Note: startup banner is printed only when the simulation actually starts (CPU path)
    
    // Record program start time
    program_start_time = std::chrono::high_resolution_clock::now();

    // Display version information
    // Keep banner plain; following lines use standardized prefixes
    std::cout << "RG-Evo: DMFE Simulation" << std::endl;
    std::cout << "=======================" << std::endl;
    std::cout << g_version_info.toString() << std::endl;
    std::cout << std::endl;

    // Subcommand: delegate grid handling to grid CLI; if handled, exit.
    {
        int gridExit = 0;
        if (dmfe::maybe_handle_grid_cli(argc, argv, gridExit)) {
            return gridExit;
        }
    }

    // Parse command line arguments
    if (!parseCommandLineArguments(argc, argv)) {
        return 1;
    }

    // Force GPU mode to false in CPU-only build
    #if !DMFE_WITH_CUDA
    if (config.gpu) {
    std::cerr << dmfe::console::WARN() << "CUDA support not compiled in. Forcing CPU mode." << std::endl;
        config.gpu = false;
    }
    #endif

    // Print startup banner only for CPU path and only when simulation starts
    if (!config.gpu) {
    std::cout << dmfe::console::INFO() << "DMFE Simulation starting (CPU mode)..." << std::endl;
    std::cout << dmfe::console::INFO() << "Available CPU cores: " << std::thread::hardware_concurrency() << std::endl;
    }

    // Initialize simulation
    init();

    // Run the main simulation
    return runSimulation();
}
