---
title: SimulationConfig

---

# SimulationConfig





## Public Attributes

|                | Name           |
| -------------- | -------------- |
| int | **[p](#variable-p)**  |
| int | **[p2](#variable-p2)**  |
| double | **[lambda](#variable-lambda)**  |
| double | **[TMCT](#variable-tmct)**  |
| double | **[T0](#variable-t0)**  |
| double | **[Gamma](#variable-gamma)**  |
| int | **[maxLoop](#variable-maxloop)**  |
| std::string | **[resultsDir](#variable-resultsdir)**  |
| std::string | **[outputDir](#variable-outputdir)**  |
| bool | **[debug](#variable-debug)**  |
| bool | **[save_output](#variable-save-output)**  |
| double | **[tmax](#variable-tmax)**  |
| double | **[delta_t_min](#variable-delta-t-min)**  |
| double | **[delta_max](#variable-delta-max)**  |
| std::vector< double > | **[rmax](#variable-rmax)**  |
| double | **[delta](#variable-delta)**  |
| double | **[delta_old](#variable-delta-old)**  |
| int | **[loop](#variable-loop)**  |
| double | **[specRad](#variable-specrad)**  |
| double | **[delta_t](#variable-delta-t)**  |
| size_t | **[len](#variable-len)**  |
| int | **[ord](#variable-ord)**  |
| bool | **[gpu](#variable-gpu)**  |
| bool | **[use_serk2](#variable-use-serk2)**  |
| int | **[sparsify_sweeps](#variable-sparsify-sweeps)**  |
| bool | **[aggressive_sparsify](#variable-aggressive-sparsify)**  |
| bool | **[async_export](#variable-async-export)**  |
| std::vector< std::string > | **[command_line_args](#variable-command-line-args)**  |
| bool | **[loaded](#variable-loaded)**  |
| std::string | **[paramDir](#variable-paramdir)**  |
| bool | **[allow_incompatible_versions](#variable-allow-incompatible-versions)**  |

## Public Attributes Documentation

### variable p

```cpp
int p = 3;
```


### variable p2

```cpp
int p2 = 12;
```


### variable lambda

```cpp
double lambda = 0.3;
```


### variable TMCT

```cpp
double TMCT = 0.805166;
```


### variable T0

```cpp
double T0 = 1e50;
```


### variable Gamma

```cpp
double Gamma = 0.0;
```


### variable maxLoop

```cpp
int maxLoop = 10000;
```


### variable resultsDir

```cpp
std::string resultsDir = "Results/";
```


### variable outputDir

```cpp
std::string outputDir = "/nobackups/jlang/Results/";
```


### variable debug

```cpp
bool debug = true;
```


### variable save_output

```cpp
bool save_output = true;
```


### variable tmax

```cpp
double tmax = 1e7;
```


### variable delta_t_min

```cpp
double delta_t_min = 1e-5;
```


### variable delta_max

```cpp
double delta_max = 1e-10;
```


### variable rmax

```cpp
std::vector< double > rmax = {3, 13, 20, 80, 180, 320, 500, 720, 980, 1280, 1620};
```


### variable delta

```cpp
double delta = 0.0;
```


### variable delta_old

```cpp
double delta_old = 0.0;
```


### variable loop

```cpp
int loop = 0;
```


### variable specRad

```cpp
double specRad = 0.0;
```


### variable delta_t

```cpp
double delta_t = 0.0;
```


### variable len

```cpp
size_t len = 512;
```


### variable ord

```cpp
int ord = 0;
```


### variable gpu

```cpp
bool gpu = true;
```


### variable use_serk2

```cpp
bool use_serk2 = true;
```


### variable sparsify_sweeps

```cpp
int sparsify_sweeps = 1;
```


### variable aggressive_sparsify

```cpp
bool aggressive_sparsify = true;
```


### variable async_export

```cpp
bool async_export = true;
```


### variable command_line_args

```cpp
std::vector< std::string > command_line_args;
```


### variable loaded

```cpp
bool loaded = false;
```


### variable paramDir

```cpp
std::string paramDir;
```


### variable allow_incompatible_versions

```cpp
bool allow_incompatible_versions = false;
```


-------------------------------

Updated on 2025-10-03 at 23:06:50 +0200