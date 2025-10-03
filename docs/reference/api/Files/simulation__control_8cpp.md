---
title: src/simulation/simulation_control.cpp

---

# src/simulation/simulation_control.cpp



## Functions

|                | Name           |
| -------------- | -------------- |
| bool | **[rollbackState](#function-rollbackstate)**(int n)<br>Roll back the simulation state by n iterations.  |

## Attributes

|                | Name           |
| -------------- | -------------- |
| SimulationConfig | **[config](#variable-config)**  |
| SimulationData * | **[sim](#variable-sim)**  |


## Functions Documentation

### function rollbackState

```cpp
bool rollbackState(
    int n
)
```

Roll back the simulation state by n iterations. 

**Parameters**: 

  * **n** Number of iterations to roll back 


**Return**: true if rollback was successful, false otherwise 

This function reduces the size of simulation vectors by removing the last n iterations, effectively rolling back the simulation state to an earlier time point.



## Attributes Documentation

### variable config

```cpp
SimulationConfig config;
```


### variable sim

```cpp
SimulationData * sim;
```



## Source code

```cpp
#include "simulation/simulation_control.hpp"
#include "simulation/simulation_data.hpp"
#include "core/config.hpp"
#include "core/config_build.hpp"
#include "interpolation/interpolation_core.hpp"
#include <iostream>

// External global variables
extern SimulationConfig config;
extern SimulationData* sim;

bool rollbackState(int n) {
#if DMFE_WITH_CUDA
    // Get current state size
    size_t currentSize = config.gpu ? sim->d_t1grid.size() : sim->h_t1grid.size();
    
    // Check if we have enough history to roll back
    if (n >= currentSize - 1) {
        std::cerr << "Error: Cannot roll back " << n << " iterations. Only " 
                  << (currentSize - 1) << " iterations available." << std::endl;
        return false;
    }
    
    // Calculate target size
    size_t targetSize = currentSize - n;
    
    if (config.gpu) {
        // Resize GPU vectors to target size
        sim->d_t1grid.resize(targetSize);
        sim->d_delta_t_ratio.resize(targetSize);
        sim->d_QKv.resize(targetSize * config.len);
        sim->d_QRv.resize(targetSize * config.len);
        sim->d_dQKv.resize(targetSize * config.len);
        sim->d_dQRv.resize(targetSize * config.len);
        sim->d_rvec.resize(targetSize);
        sim->d_drvec.resize(targetSize);
        
        // Update simulation state variables
        config.delta_t = sim->d_t1grid[targetSize-1] - sim->d_t1grid[targetSize-2];
        config.loop -= n;

        interpolateGPU();
    } else {
        // Resize host (CPU) vectors to target size via SimulationData
        sim->h_t1grid.resize(targetSize);
        sim->h_delta_t_ratio.resize(targetSize);
        sim->h_QKv.resize(targetSize * config.len);
        sim->h_QRv.resize(targetSize * config.len);
        sim->h_dQKv.resize(targetSize * config.len);
        sim->h_dQRv.resize(targetSize * config.len);
        sim->h_rvec.resize(targetSize);
        sim->h_drvec.resize(targetSize);

        // Update simulation state variables
        config.delta_t = sim->h_t1grid[targetSize-1] - sim->h_t1grid[targetSize-2];
        config.loop -= n;

        interpolate();
    }
    
    std::cout << "Successfully rolled back " << n << " iterations to time t = " 
              << (config.gpu ? sim->d_t1grid.back() : sim->h_t1grid.back()) << std::endl;
#else
    // Get current state size
    size_t currentSize = sim->h_t1grid.size();
    
    // Check if we have enough history to roll back
    if (n >= currentSize - 1) {
        std::cerr << "Error: Cannot roll back " << n << " iterations. Only " 
                  << (currentSize - 1) << " iterations available." << std::endl;
        return false;
    }
    
    // Calculate target size
    size_t targetSize = currentSize - n;
    
    // Resize host (CPU) vectors to target size via SimulationData
    sim->h_t1grid.resize(targetSize);
    sim->h_delta_t_ratio.resize(targetSize);
    sim->h_QKv.resize(targetSize * config.len);
    sim->h_QRv.resize(targetSize * config.len);
    sim->h_dQKv.resize(targetSize * config.len);
    sim->h_dQRv.resize(targetSize * config.len);
    sim->h_rvec.resize(targetSize);
    sim->h_drvec.resize(targetSize);

    // Update simulation state variables
    config.delta_t = sim->h_t1grid[targetSize-1] - sim->h_t1grid[targetSize-2];
    config.loop -= n;

    interpolate();
    
    std::cout << "Successfully rolled back " << n << " iterations to time t = " 
              << sim->h_t1grid.back() << std::endl;
#endif
    return true;
}
```


-------------------------------

Updated on 2025-10-03 at 23:06:52 +0200
