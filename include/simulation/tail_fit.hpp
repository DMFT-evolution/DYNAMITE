// Fast tail-fit and blending for QKv near theta -> 1.
// Finds the best 10-point constant fit window in the second half of theta
// for y = (1 - QK) / (1 - theta)^2, then blends the fitted form
// QK_fit = 1 - alpha * (1 - theta)^2 into the numerical solution
// for larger theta using a short smooth ramp. Designed to be called each step.

#pragma once

#include "core/config_build.hpp"

// Forward declarations of globals
struct SimulationData;
struct SimulationConfig;

// CPU path
void tailFitBlendCPU();

#if DMFE_WITH_CUDA
// GPU path
void tailFitBlendGPU();
#endif
