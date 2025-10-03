---
title: include/simulation/simulation_runner.hpp

---

# include/simulation/simulation_runner.hpp



## Functions

|                | Name           |
| -------------- | -------------- |
| int | **[runSimulation](#function-runsimulation)**()<br>Run the main simulation loop.  |
| void | **[runPerformanceBenchmark](#function-runperformancebenchmark)**()<br>Performance benchmarking for GPU operations.  |
| void | **[runPerformanceBenchmarkCPU](#function-runperformancebenchmarkcpu)**()<br>Performance benchmarking for CPU operations.  |

## Attributes

|                | Name           |
| -------------- | -------------- |
| std::chrono::high_resolution_clock::time_point | **[program_start_time](#variable-program-start-time)**  |


## Functions Documentation

### function runSimulation

```cpp
int runSimulation()
```

Run the main simulation loop. 

**Parameters**: 

  * **pool** StreamPool for GPU operations 


**Return**: int Exit code (0 for success) 

This function contains the core simulation logic including:

* File output setup
* Main time-stepping loop
* Adaptive time-step control
* Performance benchmarking (if debug mode)
* Final result output and cleanup


### function runPerformanceBenchmark

```cpp
void runPerformanceBenchmark()
```

Performance benchmarking for GPU operations. 

**Parameters**: 

  * **pool** StreamPool for GPU operations 


Runs performance tests on various GPU kernels when debug mode is enabled


### function runPerformanceBenchmarkCPU

```cpp
void runPerformanceBenchmarkCPU()
```

Performance benchmarking for CPU operations. 

Runs performance tests on various CPU functions when debug mode is enabled 



## Attributes Documentation

### variable program_start_time

```cpp
std::chrono::high_resolution_clock::time_point program_start_time;
```



## Source code

```cpp
#ifndef SIMULATION_RUNNER_HPP
#define SIMULATION_RUNNER_HPP

#include <chrono>

// Global timing variable
extern std::chrono::high_resolution_clock::time_point program_start_time;

int runSimulation();

void runPerformanceBenchmark();

void runPerformanceBenchmarkCPU();

#endif // SIMULATION_RUNNER_HPP
```


-------------------------------

Updated on 2025-10-03 at 23:06:53 +0200
