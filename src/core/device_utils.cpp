#include "core/device_utils.cuh"
#include "core/config.hpp"
#include <unistd.h>
#include <sys/resource.h>
#include <cstdlib>
#include <string>

#if defined(H5_RUNTIME_OPTIONAL)
#include "io/h5_runtime.hpp"
#endif

// External global variables for peak memory tracking
size_t peak_memory_kb = 0;
size_t peak_gpu_memory_mb = 0;

// External declaration for config global variable
extern SimulationConfig config;

bool isHDF5Available() {
#if defined(H5_RUNTIME_OPTIONAL)
	return h5rt::available();
#elif defined(USE_HDF5)
	return true;
#else
	return false;
#endif
}

// Function to get current memory usage in KB
size_t getCurrentMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // Returns KB on Linux, bytes on macOS
}

#if !DMFE_WITH_CUDA
// Function to get GPU memory usage in MB (CPU-only version always returns 0)
size_t getGPUMemoryUsage() {
    return 0;
}

// Function to get available GPU memory in MB (CPU-only version always returns 0)
size_t getAvailableGPUMemory() {
    return 0;
}

// Function to get total GPU memory in MB (CPU-only version always returns 0)
size_t getTotalGPUMemory() {
    return 0;
}

// Function to update peak memory usage
void updatePeakMemory() {
    size_t current_mem = getCurrentMemoryUsage();
    if (current_mem > peak_memory_kb) {
        peak_memory_kb = current_mem;
    }
}
#endif

// Function to get hostname
std::string getHostname() {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        return std::string(hostname);
    }
    return "unknown";
}
