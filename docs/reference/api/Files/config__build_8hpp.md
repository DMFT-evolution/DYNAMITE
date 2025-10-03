---
title: include/core/config_build.hpp

---

# include/core/config_build.hpp



## Defines

|                | Name           |
| -------------- | -------------- |
|  | **[DMFE_WITH_CUDA](#define-dmfe-with-cuda)**  |
|  | **[DMFE_DEVICE](#define-dmfe-device)**  |
|  | **[DMFE_HOST](#define-dmfe-host)**  |
|  | **[DMFE_GLOBAL](#define-dmfe-global)**  |
|  | **[DMFE_HOST_DEVICE](#define-dmfe-host-device)**  |




## Macros Documentation

### define DMFE_WITH_CUDA

```cpp
#define DMFE_WITH_CUDA 0
```


### define DMFE_DEVICE

```cpp
#define DMFE_DEVICE 
```


### define DMFE_HOST

```cpp
#define DMFE_HOST 
```


### define DMFE_GLOBAL

```cpp
#define DMFE_GLOBAL 
```


### define DMFE_HOST_DEVICE

```cpp
#define DMFE_HOST_DEVICE 
```


## Source code

```cpp
#ifndef DMFE_CONFIG_BUILD_HPP
#define DMFE_CONFIG_BUILD_HPP

// This header exposes compile-time build configuration macros.
// DMFE_WITH_CUDA is defined by CMake to indicate whether CUDA support is compiled in.

#ifndef DMFE_WITH_CUDA
#define DMFE_WITH_CUDA 0
#endif

// CUDA qualifier macros: when CUDA is enabled, use CUDA keywords; otherwise define as empty.
#if DMFE_WITH_CUDA
    #define DMFE_DEVICE __device__
    #define DMFE_HOST __host__
    #define DMFE_GLOBAL __global__
    #define DMFE_HOST_DEVICE __host__ __device__
#else
    #define DMFE_DEVICE
    #define DMFE_HOST
    #define DMFE_GLOBAL
    #define DMFE_HOST_DEVICE
#endif

#endif // DMFE_CONFIG_BUILD_HPP
```


-------------------------------

Updated on 2025-10-03 at 23:06:52 +0200
