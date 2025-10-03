---
title: include/core/gpu_memory_utils.hpp

---

# include/core/gpu_memory_utils.hpp



## Functions

|                | Name           |
| -------------- | -------------- |
| void | **[copyVectorsToGPU](#function-copyvectorstogpu)**(SimulationData & sim, size_t len) |
| void | **[copyVectorsToCPU](#function-copyvectorstocpu)**(SimulationData & sim) |
| void | **[clearAllDeviceVectors](#function-clearalldevicevectors)**(SimulationData & sim) |
| void | **[clearAllHostVectors](#function-clearallhostvectors)**(SimulationData & sim) |


## Functions Documentation

### function copyVectorsToGPU

```cpp
void copyVectorsToGPU(
    SimulationData & sim,
    size_t len
)
```


### function copyVectorsToCPU

```cpp
void copyVectorsToCPU(
    SimulationData & sim
)
```


### function clearAllDeviceVectors

```cpp
void clearAllDeviceVectors(
    SimulationData & sim
)
```


### function clearAllHostVectors

```cpp
void clearAllHostVectors(
    SimulationData & sim
)
```




## Source code

```cpp
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
```


-------------------------------

Updated on 2025-10-03 at 23:06:52 +0200
