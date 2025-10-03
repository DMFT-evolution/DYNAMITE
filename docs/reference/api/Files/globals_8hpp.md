---
title: include/core/globals.hpp

---

# include/core/globals.hpp



## Attributes

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
| double[11] | **[rmax](#variable-rmax)**  |
| double | **[delta](#variable-delta)**  |
| double | **[delta_old](#variable-delta-old)**  |
| int | **[loop](#variable-loop)**  |
| double | **[specRad](#variable-specrad)**  |
| double | **[delta_t](#variable-delta-t)**  |
| size_t | **[len](#variable-len)**  |
| int | **[ord](#variable-ord)**  |
| bool | **[gpu](#variable-gpu)**  |
| SimulationData * | **[sim](#variable-sim)**  |
| RKData * | **[rk](#variable-rk)**  |



## Attributes Documentation

### variable p

```cpp
int p;
```


### variable p2

```cpp
int p2;
```


### variable lambda

```cpp
double lambda;
```


### variable TMCT

```cpp
double TMCT;
```


### variable T0

```cpp
double T0;
```


### variable Gamma

```cpp
double Gamma;
```


### variable maxLoop

```cpp
int maxLoop;
```


### variable resultsDir

```cpp
std::string resultsDir;
```


### variable outputDir

```cpp
std::string outputDir;
```


### variable debug

```cpp
bool debug;
```


### variable save_output

```cpp
bool save_output;
```


### variable tmax

```cpp
double tmax;
```


### variable delta_t_min

```cpp
double delta_t_min;
```


### variable delta_max

```cpp
double delta_max;
```


### variable rmax

```cpp
double[11] rmax;
```


### variable delta

```cpp
double delta;
```


### variable delta_old

```cpp
double delta_old;
```


### variable loop

```cpp
int loop;
```


### variable specRad

```cpp
double specRad;
```


### variable delta_t

```cpp
double delta_t;
```


### variable len

```cpp
size_t len;
```


### variable ord

```cpp
int ord;
```


### variable gpu

```cpp
bool gpu;
```


### variable sim

```cpp
SimulationData * sim;
```


### variable rk

```cpp
RKData * rk;
```



## Source code

```cpp
// Central extern declarations for global simulation state used across modules.
#pragma once
#include <vector>
#include <string>
#include "simulation/simulation_data.hpp"
#include "EOMs/rk_data.hpp"

extern int p; 
extern int p2; 
extern double lambda; 
extern double TMCT; 
extern double T0; 
extern double Gamma; 
extern int maxLoop; 
extern std::string resultsDir; 
extern std::string outputDir; 
extern bool debug; 
extern bool save_output; 

extern double tmax; 
extern double delta_t_min; 
extern double delta_max; 
extern double rmax[11];

extern double delta; 
extern double delta_old; 
extern int loop; 
extern double specRad; 
extern double delta_t; 
extern size_t len; 
extern int ord; 
extern bool gpu; 

extern SimulationData* sim; 
extern RKData* rk; 
```


-------------------------------

Updated on 2025-10-03 at 23:06:52 +0200
