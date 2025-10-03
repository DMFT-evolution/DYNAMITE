---
title: include/io/io_utils.hpp

---

# include/io/io_utils.hpp



## Classes

|                | Name           |
| -------------- | -------------- |
| struct | **[SimulationDataSnapshot](Classes/structSimulationDataSnapshot.md)**  |
| struct | **[LoadedStateParams](Classes/structLoadedStateParams.md)**  |

## Functions

|                | Name           |
| -------------- | -------------- |
| std::vector< double > | **[importVectorFromFile](#function-importvectorfromfile)**(const std::string & filename) |
| std::vector< size_t > | **[importIntVectorFromFile](#function-importintvectorfromfile)**(const std::string & filename) |
| void | **[import](#function-import)**(SimulationData & sim, size_t len_param, int & ord_ref) |
| void | **[setupOutputDirectory](#function-setupoutputdirectory)**() |
| std::string | **[getParameterDirPath](#function-getparameterdirpath)**(const std::string & resultsDir_param, int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param, size_t len_param) |
| std::string | **[findExistingParamDir](#function-findexistingparamdir)**(const std::string & resultsDir_param, int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param, size_t len_param, double delta_t_min_param, double delta_max_param, bool use_serk2_param, bool aggressive_sparsify_param) |
| void | **[ensureDirectoryExists](#function-ensuredirectoryexists)**(const std::string & dir) |
| std::string | **[getFilename](#function-getfilename)**(const std::string & resultsDir_param, int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param, size_t len_param, double delta_t_min_param, double delta_max_param, bool use_serk2_param, bool aggressive_sparsify_param, bool save_output_param) |
| bool | **[fileExists](#function-fileexists)**(const std::string & filename) |
| bool | **[loadSimulationStateBinary](#function-loadsimulationstatebinary)**(const std::string & filename, SimulationData & sim, int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param, size_t len_param, double delta_t_min_param, double delta_max_param, bool use_serk2_param, bool aggressive_sparsify_param, LoadedStateParams & loaded_params) |
| bool | **[checkParametersMatch](#function-checkparametersmatch)**(const std::string & paramFilename, int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param, size_t len_param, double delta_t_min_param, double delta_max_param, bool use_serk2_param, bool aggressive_sparsify_param) |
| bool | **[loadSimulationState](#function-loadsimulationstate)**(const std::string & filename, SimulationData & sim, int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param, size_t len_param, double delta_t_min_param, double delta_max_param, bool use_serk2_param, bool aggressive_sparsify_param, LoadedStateParams & loaded_params) |
| void | **[saveHistory](#function-savehistory)**(const std::string & filename, double delta, double delta_t, SimulationData & simulation, size_t len_param, double T0_param, bool gpu_param) |
| void | **[saveParametersToFile](#function-saveparameterstofile)**(const std::string & dirPath, double delta, double delta_t) |
| void | **[saveSimulationStateBinary](#function-savesimulationstatebinary)**(const std::string & filename, double delta, double delta_t) |
| SimulationDataSnapshot | **[saveSimulationState](#function-savesimulationstate)**(const std::string & filename, double delta, double delta_t) |
| void | **[saveCompressedData](#function-savecompresseddata)**(const std::string & dirPath) |
| void | **[waitForAsyncSavesToComplete](#function-waitforasyncsavestocomplete)**() |
| SimulationDataSnapshot | **[createDataSnapshot](#function-createdatasnapshot)**() |
| void | **[saveHistoryAsync](#function-savehistoryasync)**(const std::string & filename, double delta, double delta_t, const SimulationDataSnapshot & snapshot) |
| void | **[saveCompressedDataAsync](#function-savecompresseddataasync)**(const std::string & dirPath, const SimulationDataSnapshot & snapshot) |
| std::string | **[getCurrentTimestamp](#function-getcurrenttimestamp)**() |
| double | **[getRuntimeSeconds](#function-getruntimeseconds)**() |
| std::string | **[formatDuration](#function-formatduration)**(double total_seconds) |
| std::string | **[getGPUInfo](#function-getgpuinfo)**() |
| std::string | **[formatMemory](#function-formatmemory)**(size_t memory_kb) |


## Functions Documentation

### function importVectorFromFile

```cpp
std::vector< double > importVectorFromFile(
    const std::string & filename
)
```


### function importIntVectorFromFile

```cpp
std::vector< size_t > importIntVectorFromFile(
    const std::string & filename
)
```


### function import

```cpp
void import(
    SimulationData & sim,
    size_t len_param,
    int & ord_ref
)
```


### function setupOutputDirectory

```cpp
void setupOutputDirectory()
```


### function getParameterDirPath

```cpp
std::string getParameterDirPath(
    const std::string & resultsDir_param,
    int p_param,
    int p2_param,
    double lambda_param,
    double T0_param,
    double Gamma_param,
    size_t len_param
)
```


### function findExistingParamDir

```cpp
std::string findExistingParamDir(
    const std::string & resultsDir_param,
    int p_param,
    int p2_param,
    double lambda_param,
    double T0_param,
    double Gamma_param,
    size_t len_param,
    double delta_t_min_param,
    double delta_max_param,
    bool use_serk2_param,
    bool aggressive_sparsify_param
)
```


### function ensureDirectoryExists

```cpp
void ensureDirectoryExists(
    const std::string & dir
)
```


### function getFilename

```cpp
std::string getFilename(
    const std::string & resultsDir_param,
    int p_param,
    int p2_param,
    double lambda_param,
    double T0_param,
    double Gamma_param,
    size_t len_param,
    double delta_t_min_param,
    double delta_max_param,
    bool use_serk2_param,
    bool aggressive_sparsify_param,
    bool save_output_param
)
```


### function fileExists

```cpp
bool fileExists(
    const std::string & filename
)
```


### function loadSimulationStateBinary

```cpp
bool loadSimulationStateBinary(
    const std::string & filename,
    SimulationData & sim,
    int p_param,
    int p2_param,
    double lambda_param,
    double T0_param,
    double Gamma_param,
    size_t len_param,
    double delta_t_min_param,
    double delta_max_param,
    bool use_serk2_param,
    bool aggressive_sparsify_param,
    LoadedStateParams & loaded_params
)
```


### function checkParametersMatch

```cpp
bool checkParametersMatch(
    const std::string & paramFilename,
    int p_param,
    int p2_param,
    double lambda_param,
    double T0_param,
    double Gamma_param,
    size_t len_param,
    double delta_t_min_param,
    double delta_max_param,
    bool use_serk2_param,
    bool aggressive_sparsify_param
)
```


### function loadSimulationState

```cpp
bool loadSimulationState(
    const std::string & filename,
    SimulationData & sim,
    int p_param,
    int p2_param,
    double lambda_param,
    double T0_param,
    double Gamma_param,
    size_t len_param,
    double delta_t_min_param,
    double delta_max_param,
    bool use_serk2_param,
    bool aggressive_sparsify_param,
    LoadedStateParams & loaded_params
)
```


### function saveHistory

```cpp
void saveHistory(
    const std::string & filename,
    double delta,
    double delta_t,
    SimulationData & simulation,
    size_t len_param,
    double T0_param,
    bool gpu_param
)
```


### function saveParametersToFile

```cpp
void saveParametersToFile(
    const std::string & dirPath,
    double delta,
    double delta_t
)
```


### function saveSimulationStateBinary

```cpp
void saveSimulationStateBinary(
    const std::string & filename,
    double delta,
    double delta_t
)
```


### function saveSimulationState

```cpp
SimulationDataSnapshot saveSimulationState(
    const std::string & filename,
    double delta,
    double delta_t
)
```


### function saveCompressedData

```cpp
void saveCompressedData(
    const std::string & dirPath
)
```


### function waitForAsyncSavesToComplete

```cpp
void waitForAsyncSavesToComplete()
```


### function createDataSnapshot

```cpp
SimulationDataSnapshot createDataSnapshot()
```


### function saveHistoryAsync

```cpp
void saveHistoryAsync(
    const std::string & filename,
    double delta,
    double delta_t,
    const SimulationDataSnapshot & snapshot
)
```


### function saveCompressedDataAsync

```cpp
void saveCompressedDataAsync(
    const std::string & dirPath,
    const SimulationDataSnapshot & snapshot
)
```


### function getCurrentTimestamp

```cpp
std::string getCurrentTimestamp()
```


### function getRuntimeSeconds

```cpp
double getRuntimeSeconds()
```


### function formatDuration

```cpp
std::string formatDuration(
    double total_seconds
)
```


### function getGPUInfo

```cpp
std::string getGPUInfo()
```


### function formatMemory

```cpp
std::string formatMemory(
    size_t memory_kb
)
```




## Source code

```cpp
#pragma once

#include <string>
#include <vector>
#include <chrono>
#include "simulation/simulation_data.hpp"
#include "core/config.hpp"

// Struct to hold simulation data snapshot for background saving
struct SimulationDataSnapshot {
    std::vector<double> QKv, QRv, dQKv, dQRv, t1grid, rvec, drvec;
    std::vector<double> QKB1int, QRB1int, theta;
    double energy;
    double t_current;
    int current_len, current_loop;
    SimulationConfig config_snapshot;
    
    // Version and memory info for async saving
    std::string code_version, git_hash, git_branch, git_tag, build_date, build_time, compiler_version, cuda_version;
    bool git_dirty;
    size_t peak_memory_kb_snapshot, peak_gpu_memory_mb_snapshot;
    std::chrono::high_resolution_clock::time_point program_start_time_snapshot;
};

// Structure to hold loaded simulation state parameters
struct LoadedStateParams {
    double delta, delta_t;
    int loop;
};

// Basic text file imports
std::vector<double> importVectorFromFile(const std::string& filename);
std::vector<size_t> importIntVectorFromFile(const std::string& filename);
void import(SimulationData& sim, size_t len_param, int& ord_ref); // bulk grid import

// Path helpers
void setupOutputDirectory();
std::string getParameterDirPath(const std::string& resultsDir_param, int p_param, int p2_param, 
                               double lambda_param, double T0_param, double Gamma_param, size_t len_param);
std::string findExistingParamDir(const std::string& resultsDir_param, int p_param, int p2_param,
                                double lambda_param, double T0_param, double Gamma_param, size_t len_param,
                                double delta_t_min_param, double delta_max_param, bool use_serk2_param, bool aggressive_sparsify_param);
void ensureDirectoryExists(const std::string& dir);
std::string getFilename(const std::string& resultsDir_param, int p_param, int p2_param, 
                       double lambda_param, double T0_param, double Gamma_param, size_t len_param, 
                       double delta_t_min_param, double delta_max_param, bool use_serk2_param, bool aggressive_sparsify_param,
                       bool save_output_param);

// File existence
bool fileExists(const std::string& filename);

// Loaders
bool loadSimulationStateBinary(const std::string& filename, SimulationData& sim,
                              int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param,
                              size_t len_param, double delta_t_min_param, double delta_max_param,
                              bool use_serk2_param, bool aggressive_sparsify_param,
                              LoadedStateParams& loaded_params);
#if defined(USE_HDF5)
bool loadSimulationStateHDF5(const std::string& filename, SimulationData& sim,
                            int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param,
                            size_t len_param, double delta_t_min_param, double delta_max_param,
                            bool use_serk2_param, bool aggressive_sparsify_param,
                            LoadedStateParams& loaded_params);
#endif
bool checkParametersMatch(const std::string& paramFilename, int p_param, int p2_param, double lambda_param, 
                         double T0_param, double Gamma_param, size_t len_param, double delta_t_min_param, double delta_max_param,
                         bool use_serk2_param, bool aggressive_sparsify_param);
bool loadSimulationState(const std::string& filename, SimulationData& sim,
                        int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param, 
                        size_t len_param, double delta_t_min_param, double delta_max_param,
                        bool use_serk2_param, bool aggressive_sparsify_param,
                        LoadedStateParams& loaded_params);

// History saving
void saveHistory(const std::string& filename, double delta, double delta_t, 
                 SimulationData& simulation, size_t len_param, double T0_param, bool gpu_param);

// Save functions
void saveParametersToFile(const std::string& dirPath, double delta, double delta_t);
void saveSimulationStateBinary(const std::string& filename, double delta, double delta_t);
#if defined(USE_HDF5)
void saveSimulationStateHDF5(const std::string& filename, double delta, double delta_t);
#endif
SimulationDataSnapshot saveSimulationState(const std::string& filename, double delta, double delta_t);
void saveCompressedData(const std::string& dirPath);
void waitForAsyncSavesToComplete();

// GPU-side internal functions (defined in .cu files, called by .cpp)
SimulationDataSnapshot createDataSnapshot();
void saveHistoryAsync(const std::string& filename, double delta, double delta_t, const SimulationDataSnapshot& snapshot);
void saveCompressedDataAsync(const std::string& dirPath, const SimulationDataSnapshot& snapshot);

// Utility functions for save operations
std::string getCurrentTimestamp();
double getRuntimeSeconds();
std::string formatDuration(double total_seconds);
std::string getGPUInfo();
std::string formatMemory(size_t memory_kb);

// Helper functions for computation (used by saveHistory and other functions)
```


-------------------------------

Updated on 2025-10-03 at 23:06:53 +0200
