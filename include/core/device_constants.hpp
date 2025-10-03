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
