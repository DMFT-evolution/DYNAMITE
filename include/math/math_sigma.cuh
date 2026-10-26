#pragma once
#include "math/math_sigma.hpp"

#if DMFE_WITH_CUDA
#include "core/device_vector.hpp"

// GPU kernel and host wrappers for SigmaK/SigmaR evaluations
__global__ void computeSigmaKandRKernel(const double* __restrict__ qK,
										const double* __restrict__ qR,
										double* __restrict__ sigmaK,
										double* __restrict__ sigmaR,
										size_t len);

// ============================================================================
// [PERF] Batched sigma evaluation.
//
// interpolateGPU() evaluates SigmaK/SigmaR on four independent L x L slices
// (A1, A2, B1, B2), one kernel launch each, four times per Runge-Kutta stage.
// Each launch is a pure element-wise map, so the four can be merged into a
// single launch with a (blocks x n) grid.
//
// BIT-EXACTNESS: element i of slice k is computed by exactly the same expression
// from exactly the same two inputs as before -- eval_sigmaK_sigmaR_fused is
// unchanged and there is no reduction, no shared state and no cross-element
// dependency.  Only the block-to-launch mapping changes.
// ============================================================================
struct SigmaBatchDesc {
    static constexpr int MAX = 4;
    const double* qK[MAX];
    const double* qR[MAX];
    double*       sigmaK[MAX];
    double*       sigmaR[MAX];
    int           n = 0;

    void add(const double* qK_, const double* qR_, double* sK_, double* sR_) {
        qK[n] = qK_; qR[n] = qR_; sigmaK[n] = sK_; sigmaR[n] = sR_; ++n;
    }
};

void computeSigmaKandRBatchedGPU(const SigmaBatchDesc& batch, size_t len, cudaStream_t stream);

// Sigma GPU function declarations
void SigmaKGPU(const dmfe::device_vector<double>& qk, dmfe::device_vector<double>& result, cudaStream_t stream = 0);
void SigmaRGPU(const dmfe::device_vector<double>& qk, const dmfe::device_vector<double>& qr, dmfe::device_vector<double>& result, cudaStream_t stream = 0);
#endif // DMFE_WITH_CUDA
