#pragma once
// ============================================================================
// [PERF] Host-side mirror of sim->device->t1grid.back().
//
// t1grid lives in device memory, so every `sim->device->t1grid.back()` is a
// thrust::device_reference conversion: a one-element blocking device-to-host
// copy, which drains the whole CUDA queue.  The profile of a 1124-step run shows
// ~10 such blocking copies per step, 0.89 s of wall time spent inside
// cudaMemcpyAsync, and a GPU that is idle 40% of the time waiting for the host to
// issue the next batch of launches.
//
// The last element of t1grid is, however, always a value the host itself
// produced: appendAllGPU_ptr and replaceAllGPU_ptr pass `t` to append4Kernel /
// replace4Kernel, which store exactly that double at t1grid[idx].  Caching it on
// the host therefore returns the *same bit pattern*, not an approximation.
//
// BIT-EXACTNESS: t1LastSet() is called with the very double that the kernel
// stores, so a cache hit is indistinguishable from the copy it replaces.  Every
// other route by which t1grid can change (rollback, sparsification, state
// loading, clearing) calls t1LastInvalidate(), after which the next query falls
// back to reading the device.  A missed invalidation would change results and is
// caught by the bit-for-bit regression check.
// ============================================================================
#include "core/config_build.hpp"

#if DMFE_WITH_CUDA
namespace dmfe {

// Value of sim->device->t1grid.back(), read from the device only when the cache
// is cold.
double t1LastGPU();

// Record the double that has just been written into t1grid[last] by a kernel.
void t1LastSet(double v);

// Any other modification of t1grid (resize, gather, reload, clear).
void t1LastInvalidate();

} // namespace dmfe
#endif // DMFE_WITH_CUDA
