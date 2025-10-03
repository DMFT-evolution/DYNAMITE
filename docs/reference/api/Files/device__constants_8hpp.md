---
title: include/core/device_constants.hpp

---

# include/core/device_constants.hpp



## Attributes

|                | Name           |
| -------------- | -------------- |
| __constant__ int | **[d_p](#variable-d-p)**  |
| __constant__ int | **[d_p2](#variable-d-p2)**  |
| __constant__ double | **[d_lambda](#variable-d-lambda)**  |



## Attributes Documentation

### variable d_p

```cpp
__constant__ int d_p;
```


### variable d_p2

```cpp
__constant__ int d_p2;
```


### variable d_lambda

```cpp
__constant__ double d_lambda;
```



## Source code

```cpp
// Centralized CUDA constant memory declarations.
//
// These are defined exactly once in src/core/device_constants.cu and
// referenced everywhere else via these extern declarations. The previous
// macro-based scheme caused NVCC (without relocatable device code) to treat
// each "extern" as a separate static definition, leading to zero-initialized
// duplicates visible to different translation units. Keeping only extern
// declarations here plus enabling CUDA separable compilation fixes that.
//
// NOTE: Ensure the targets using these symbols have CUDA_SEPARABLE_COMPILATION
// enabled in CMakeLists.txt so a device linking step unifies the symbols.
#pragma once

extern __constant__ int d_p;
extern __constant__ int d_p2;
extern __constant__ double d_lambda;
```


-------------------------------

Updated on 2025-10-03 at 23:06:52 +0200
