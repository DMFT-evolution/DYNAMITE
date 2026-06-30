#pragma once

#include "core/config_build.hpp"
#include "core/console.hpp"
#include <string>
#include <cstddef>

// Device capability check
bool isCompatibleGPUInstalled();


// System utility functions
bool isHDF5Available();
size_t getCurrentMemoryUsageMB();
// Total physical system memory in MB (Linux); returns 0 if unavailable
size_t getTotalSystemMemoryMB();
size_t getGPUMemoryUsage();
size_t getAvailableGPUMemory();
void updatePeakMemory();
std::string getHostname();

// External global variables for peak memory tracking
extern size_t peak_memory_mb;
extern size_t peak_gpu_memory_mb;