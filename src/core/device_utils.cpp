#include "core/device_utils.hpp"
#include "core/config.hpp"
#include <unistd.h>
#include <sys/resource.h>
#include <cstdlib>
#include <string>

#if defined(H5_RUNTIME_OPTIONAL)
#include "io/h5_runtime.hpp"
#endif

// External global variables for peak memory tracking
size_t peak_memory_mb = 0;
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

// Function to get current memory usage in MB (rounded down, minimum 1 when non-zero)
size_t getCurrentMemoryUsageMB() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
#if defined(__APPLE__)
    size_t bytes = static_cast<size_t>(usage.ru_maxrss);
    size_t mb = bytes / (1024ULL * 1024ULL);
    if (mb == 0 && bytes > 0) mb = 1; // ensure non-zero when memory is in use
    return mb;
#else
    size_t kb = static_cast<size_t>(usage.ru_maxrss);
    size_t mb = kb / 1024ULL;
    if (mb == 0 && kb > 0) mb = 1;
    return mb;
#endif
}

// [CPC-REV] <sys/sysinfo.h> is Linux-only; the unconditional include broke the
// advertised macOS CPU build (referee 2, point 5). Query hw.memsize on Darwin and
// keep a portable sysconf() fallback.
#if defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif
#include <cstdint>

// Function to get total physical system memory in MB. Returns 0 if unavailable.
size_t getTotalSystemMemoryMB() {
#if defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        // totalram is in units of mem_unit bytes
        unsigned long long bytes = static_cast<unsigned long long>(info.totalram) * info.mem_unit;
        return static_cast<size_t>(bytes / (1024ULL * 1024ULL));
    }
#elif defined(__APPLE__)
    uint64_t memsize = 0;
    size_t memsize_len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &memsize_len, nullptr, 0) == 0 && memsize > 0) {
        return static_cast<size_t>(memsize / (1024ULL * 1024ULL));
    }
#endif
    // Portable POSIX fallback (Linux and macOS)
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) {
        unsigned long long bytes = static_cast<unsigned long long>(pages) * static_cast<unsigned long long>(page_size);
        return static_cast<size_t>(bytes / (1024ULL * 1024ULL));
    }
#endif
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
    size_t current_mem = getCurrentMemoryUsageMB();
    if (current_mem > peak_memory_mb) {
        peak_memory_mb = current_mem;
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
