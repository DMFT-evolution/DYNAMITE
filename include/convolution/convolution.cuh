#pragma once
#include "convolution/convolution.hpp"

#if DMFE_WITH_CUDA
#include "core/device_vector.hpp"
#include <thrust/device_ptr.h>
#include <cuda_runtime.h>

// ============================================================================
// [PERF] Batched convolution.
//
// QKRstepGPU issues eight independent ConvA/ConvR reductions per Runge-Kutta
// stage, each as its own kernel launch of `depth` blocks x 64 threads.  With
// depth = L = 512 that is only 1024 warps per launch, far too few to hide memory
// latency on an H100, and it costs eight launches (~2.3 us of host time each) per
// stage.  ConvBatchedGPU runs all of them in a single launch with a
// (depth x n) grid.
//
// BIT-EXACTNESS: each block performs exactly the same computation as before --
// same blockDim, same shared-memory tree reduction, same operand order (note that
// ConvA accumulates integ*f*g while ConvR accumulates f*g*integ, and the two are
// NOT interchangeable in floating point, so the two code paths are kept separate),
// and the same final scaling.  Only the mapping of blocks to launches changes.
// ============================================================================
enum class ConvKind : int { A = 0, R = 1 };

struct ConvBatchDesc {
    static constexpr int MAX = 12;
    const double* f[MAX];
    const double* g[MAX];
    double*       out[MAX];
    int           kind[MAX];
    int           n = 0;

    void add(ConvKind k, const double* f_, const double* g_, double* out_) {
        f[n] = f_; g[n] = g_; out[n] = out_; kind[n] = static_cast<int>(k); ++n;
    }
};

// ============================================================================
// [PERF] Batched ConvA with a host-side scalar t.
//
// rstepGPU issues two ConvA reductions per Runge-Kutta stage and drstep2GPU six
// more per accepted step; with the shipped grids each of them is a *single* block
// of 64 threads, so the launches cost far more than the work.  They are merged
// into one launch with a (depth x n) grid.
//
// BIT-EXACTNESS: the body is ConvAGPUKernel(scalar-t overload) character for
// character -- same blockDim, same shared-memory tree reduction, same operand
// order, same final scaling, including the (depth == 1 ? 1.0 : theta[j]) case.
// Only the mapping of blocks to launches changes.  Entries of one batch must
// share `length` and `depth`, which every call site does.
// ============================================================================
struct ConvAScalarBatchDesc {
    static constexpr int MAX = 8;
    const double* f[MAX];
    const double* g[MAX];
    double*       out[MAX];
    double        t[MAX];
    int           n = 0;

    void add(const double* f_, const double* g_, double* out_, double t_) {
        f[n] = f_; g[n] = g_; out[n] = out_; t[n] = t_; ++n;
    }
};

void ConvABatchedScalarGPU(const ConvAScalarBatchDesc& batch,
                           const dmfe::device_vector<double>& integ,
                           const dmfe::device_vector<double>& theta,
                           size_t length,
                           size_t depth,
                           cudaStream_t stream);

void ConvBatchedGPU(const ConvBatchDesc& batch,
                    const dmfe::device_vector<double>& integ,
                    const dmfe::device_vector<double>& theta,
                    const dmfe::device_vector<double>& t,
                    size_t length,
                    size_t depth,
                    cudaStream_t stream);

// ConvA (QK-style) GPU interfaces
dmfe::device_vector<double> ConvAGPU(const dmfe::device_vector<double>& f,
                                       const dmfe::device_vector<double>& g,
                                       double t,
                                       const dmfe::device_vector<double>& integ,
                                       const dmfe::device_vector<double>& theta,
                                       cudaStream_t stream = 0);

dmfe::device_vector<double> ConvAGPU(const dmfe::device_vector<double>& f,
                                       const thrust::device_ptr<double>& g,
                                       double t,
                                       const dmfe::device_vector<double>& integ,
                                       const dmfe::device_vector<double>& theta,
                                       cudaStream_t stream = 0);

void ConvAGPU_Stream(const dmfe::device_vector<double>& f,
                     const dmfe::device_vector<double>& g,
                     dmfe::device_vector<double>& out,
                     dmfe::device_vector<double>& t,
                     const dmfe::device_vector<double>& integ,
                     const dmfe::device_vector<double>& theta,
                     cudaStream_t stream = 0);

void ConvAGPU_Stream(const dmfe::device_vector<double>& f,
                     const thrust::device_ptr<double>& g,
                     dmfe::device_vector<double>& out,
                     double t,
                     const dmfe::device_vector<double>& integ,
                     const dmfe::device_vector<double>& theta,
                     cudaStream_t stream = 0);

// Const-pointer overload for ConvA stream variant
void ConvAGPU_Stream(const dmfe::device_vector<double>& f,
                     const thrust::device_ptr<const double>& g,
                     dmfe::device_vector<double>& out,
                     double t,
                     const dmfe::device_vector<double>& integ,
                     const dmfe::device_vector<double>& theta,
                     cudaStream_t stream = 0);

// ConvR (QR-style) GPU interfaces
dmfe::device_vector<double> ConvRGPU(const dmfe::device_vector<double>& f,
                                       const dmfe::device_vector<double>& g,
                                       double t,
                                       const dmfe::device_vector<double>& integ,
                                       const dmfe::device_vector<double>& theta,
                                       cudaStream_t stream = 0);

void ConvRGPU_Stream(const dmfe::device_vector<double>& f,
                     const dmfe::device_vector<double>& g,
                     dmfe::device_vector<double>& out,
                     const dmfe::device_vector<double>& t,
                     const dmfe::device_vector<double>& integ,
                     const dmfe::device_vector<double>& theta,
                     cudaStream_t stream = 0);
#endif
