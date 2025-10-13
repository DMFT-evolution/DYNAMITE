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

#include <sys/sysinfo.h>
// Function to get total physical system memory in KB (Linux). Returns 0 if unavailable.
size_t getTotalSystemMemoryKB() {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        // totalram is in units of mem_unit bytes
        unsigned long long bytes = static_cast<unsigned long long>(info.totalram) * info.mem_unit;
        return static_cast<size_t>(bytes / 1024ULL);
    }
    // Fallback using sysconf
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        unsigned long long bytes = static_cast<unsigned long long>(pages) * static_cast<unsigned long long>(page_size);
        return static_cast<size_t>(bytes / 1024ULL);
    }
    return 0;
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
