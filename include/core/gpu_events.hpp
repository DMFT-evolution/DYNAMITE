#pragma once
// ============================================================================
// [PERF] Long-lived CUDA events used to express cross-stream dependencies that
// were previously expressed by device-wide barriers (cudaDeviceSynchronize) or,
// worse, by the implicit barrier of a one-element device-to-host copy.
//
// BIT-EXACTNESS: an event constrains *when* a kernel may start, never *what* it
// computes.  No kernel is added, removed, re-parameterised or re-ordered with
// respect to another kernel that touches the same data; only the mechanism that
// enforces the ordering changes.  Every stored double is therefore unchanged.
// ============================================================================
#include "core/config_build.hpp"

#if DMFE_WITH_CUDA
#include <cuda_runtime.h>
#include <cstddef>

namespace dmfe {

// Named slots, so that producer and consumer cannot drift apart.
enum : std::size_t {
    EV_INTERP_S1   = 0,  // interpolateGPU chain on pool[1] (B2 slice)
    EV_INTERP_S2   = 1,  // interpolateGPU chain on pool[2] (A1 slice)
    EV_INTERP_S3   = 2,  // interpolateGPU chain on pool[3] (A2 slice)
    EV_RSTEP_OUT   = 3,  // rstepGPU: r(t) written into rvec[idx] on pool[3]
    EV_BSEARCH     = 4,  // interpolateGPU: posB1xOld and posB2xOld ready on pool[0]
    EV_COUNT       = 5
};

class EventPool {
public:
    static EventPool& instance() { static EventPool p; return p; }
    cudaEvent_t operator[](std::size_t i) const { return ev_[i]; }
private:
    EventPool() {
        for (std::size_t i = 0; i < EV_COUNT; ++i)
            cudaEventCreateWithFlags(&ev_[i], cudaEventDisableTiming);
    }
    ~EventPool() {
        for (std::size_t i = 0; i < EV_COUNT; ++i) cudaEventDestroy(ev_[i]);
    }
    EventPool(const EventPool&) = delete;
    EventPool& operator=(const EventPool&) = delete;
    cudaEvent_t ev_[EV_COUNT];
};

} // namespace dmfe
#endif // DMFE_WITH_CUDA
