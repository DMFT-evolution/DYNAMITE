#include <chrono>
// Single definition of global program_start_time used across I/O and telemetry modules.
// Separated to prevent accidental multiple definitions and to keep nvcc from compiling
// unrelated UI logic as CUDA when source classification shifts.
std::chrono::high_resolution_clock::time_point program_start_time;
