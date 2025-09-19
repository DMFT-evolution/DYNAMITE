#ifndef SIMULATION_RUNNER_HPP
#define SIMULATION_RUNNER_HPP

#include <chrono>

// Global timing variable
extern std::chrono::high_resolution_clock::time_point program_start_time;

/**
 * @brief Run the main simulation loop
 * 
 * This function contains the core simulation logic including:
 * - File output setup
 * - Main time-stepping loop
 * - Adaptive time-step control
 * - Performance benchmarking (if debug mode)
 * - Final result output and cleanup
 * 
 * @param pool StreamPool for GPU operations
 * @return int Exit code (0 for success)
 */
int runSimulation();

/**
 * @brief Performance benchmarking for GPU operations
 * 
 * Runs performance tests on various GPU kernels when debug mode is enabled
 * 
 * @param pool StreamPool for GPU operations
 */
void runPerformanceBenchmark();

/**
 * @brief Performance benchmarking for CPU operations
 * 
 * Runs performance tests on various CPU functions when debug mode is enabled
 */
void runPerformanceBenchmarkCPU();

#endif // SIMULATION_RUNNER_HPP
