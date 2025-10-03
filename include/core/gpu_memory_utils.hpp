#pragma once
#include "core/config_build.hpp"
#include <vector>
#include <cstddef>

#if DMFE_WITH_CUDA
#include <cuda_runtime.h>
#endif

struct SimulationData; // forward

// Copy host -> device using SimulationData host (h_*) into device (d_*) members.
void copyVectorsToGPU(SimulationData& sim, size_t len);

// Copy device -> host (d_* -> h_*)
void copyVectorsToCPU(SimulationData& sim);

// Clear all device vectors
void clearAllDeviceVectors(SimulationData& sim);

// Clear host mirrors
void clearAllHostVectors(SimulationData& sim);

#if DMFE_WITH_CUDA
// Copy constants to device constant memory
void copyParametersToDevice(int p_host, int p2_host, double lambda_host);

// Raw utility allocators (caller frees with cudaFree)
double* copyVectorToDeviceRaw(const std::vector<double>& host_vec);
size_t* copyVectorToDeviceRaw(const std::vector<size_t>& host_vec);
#endif

