#include "EOMs/runge_kutta.cuh"
#include "core/t1_cache.hpp"
#include "core/globals.hpp"
#include "EOMs/device_rk_data.hpp"

#include "core/config.hpp"
#include "math/math_ops.hpp"
#include "core/vector_utils.cuh"
#include "convolution/convolution.cuh"
#include "core/compute_utils.hpp"
#include "EOMs/time_steps_cuda.cuh"
#include "interpolation/interpolation_core.hpp"
#include "simulation/device_simulation_data.hpp"
#include "core/device_utils.cuh"
#include "math/math_sigma.cuh"
#include <cuda_runtime.h>
#include "core/device_vector.hpp"
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include "core/console.hpp"
#include <cmath>
#include <stdexcept>

using namespace std;

// External declarations for global variables
extern SimulationConfig config;
extern SimulationData* sim;
extern RKData* rk;

// [CPC-REV] double-precision atomic max, needed by the extended error reduction.
__device__ __forceinline__ void ATOMIC_MAX_DBL(double* addr, double val)
{
    unsigned long long int* a = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old = *a, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= val) break;
        old = atomicCAS(a, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

// Kernel implementations
__global__ void computeError(
    const double* __restrict__ gKfinal,
    const double* __restrict__ gKe,
    const double* __restrict__ gRfinal,
    const double* __restrict__ gRe,
    double* __restrict__ result,
    size_t len)
{
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // [CPC-REV] The reduction now carries three quantities instead of one, so that
    // the host can form any of the norms selectable with --error-norm:
    //   result[0] = sum_i (|dC_i| + |dR_i|)      (extensive 1-norm, legacy default)
    //   result[1] = sum_i (dC_i^2 + dR_i^2)      (for the RMS norm)
    //   result[2] = max_i (|dC_i|, |dR_i|)       (for the maximum norm)
    // result[0] is bit-identical to the previous single-value reduction.
    double local_sum = 0.0, local_sq = 0.0, local_max = 0.0;

    // Process both K and R errors in one pass
    if (i < len) {
        const double dc = fabs(gKfinal[i] - gKe[i]);
        const double dr = fabs(gRfinal[i] - gRe[i]);
        local_sum += dc;
        local_sum += dr;
        local_sq   = dc * dc + dr * dr;
        local_max  = fmax(dc, dr);
    }

    double* ssum = sdata;
    double* ssq  = sdata + blockDim.x;
    double* smax = sdata + 2 * blockDim.x;
    ssum[tid] = local_sum;
    ssq[tid]  = local_sq;
    smax[tid] = local_max;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
            ssq[tid]  += ssq[tid + s];
            smax[tid]  = fmax(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        ATOMIC_ADD_DBL(result,     ssum[0]);
        ATOMIC_ADD_DBL(result + 1, ssq[0]);
        ATOMIC_MAX_DBL(result + 2, smax[0]);
    }
}

// [CPC-REV] Host-side assembly of the norm selected with --error-norm.
// See rk_error_norm() in runge_kutta.cpp for the CPU counterpart.
static inline double dmfe_rk_error_from_reduction(size_t len)
{
    const double N = static_cast<double>(2 * len);
    switch (config.error_norm) {
        case 1:  return sim->device->error_result[0] / N;
        case 2:  return sqrt(sim->device->error_result[1] / N);
        case 3:  return sim->device->error_result[2];
        default: return sim->device->error_result[0];
    }
}

// ============================================================================
// [PERF] Stage-0 broadcast.
//
// Each Runge-Kutta entry point used to issue eight thrust::copy calls (the SAME
// source slice copied into gK/gK0/gKfinal/gKe and gR/gR0/gRfinal/gRe) plus a
// thrust::fill of error_result.  thrust's default policy is synchronous, so those
// were nine kernel launches AND nine host-side stream synchronizations per step.
// One kernel reads each source once and writes all eight destinations.
//
// BIT-EXACTNESS: pure data movement -- every destination receives exactly the
// same double it received before -- plus a fill with 0.0.  No arithmetic at all.
// Launched on the legacy default stream, exactly like the thrust calls it
// replaces, so the implicit ordering against the blocking pool streams is
// unchanged.
// ============================================================================
__global__ void rkBroadcastInitKernel(const double* __restrict__ qk,
                                      const double* __restrict__ qr,
                                      double* __restrict__ gK,  double* __restrict__ gK0,
                                      double* __restrict__ gKf, double* __restrict__ gKe,
                                      double* __restrict__ gR,  double* __restrict__ gR0,
                                      double* __restrict__ gRf, double* __restrict__ gRe,
                                      double* __restrict__ err, size_t nerr, size_t len)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        const double k = qk[i];
        const double r = qr[i];
        gK[i] = k; gK0[i] = k; gKf[i] = k; gKe[i] = k;
        gR[i] = r; gR0[i] = r; gRf[i] = r; gRe[i] = r;
    }
    if (i < nerr) err[i] = 0.0;
}

// ============================================================================
// [PERF] Stage-0 snapshot h -> h0.
//
// Each Runge-Kutta driver copied hK and hR into hK0 and hR0 with
// thrust::device_vector::assign (or operator=).  thrust's default execution
// policy is synchronous, so those two copies were two more host-side stream
// synchronisations per step -- and the host, not the GPU, is what limits this
// code.  One kernel moves both arrays.
//
// BIT-EXACTNESS: pure data movement.  Every destination element receives exactly
// the double that the thrust copy put there; there is no arithmetic.  The launch
// is on the legacy default stream, exactly like the thrust calls it replaces, so
// the implicit ordering against the blocking pool streams is unchanged.
// ============================================================================
__global__ void rkCopyH0Kernel(const double* __restrict__ hK,
                               const double* __restrict__ hR,
                               double* __restrict__ hK0,
                               double* __restrict__ hR0,
                               size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        hK0[i] = hK[i];
        hR0[i] = hR[i];
    }
}

static inline void rkCopy2Both(const double* srcA, const double* srcB,
                               double* dstA, double* dstB, size_t n)
{
    const int threads = 64;
    const int blocks  = static_cast<int>((n + threads - 1) / threads);
    rkCopyH0Kernel<<<blocks, threads>>>(srcA, srcB, dstA, dstB, n);
}

static inline void rkCopyH0Both(size_t n)
{
    rkCopy2Both(thrust::raw_pointer_cast(rk->device->hK.data()),
                thrust::raw_pointer_cast(rk->device->hR.data()),
                thrust::raw_pointer_cast(rk->device->hK0.data()),
                thrust::raw_pointer_cast(rk->device->hR0.data()),
                n);
}

static inline void rkBroadcastInit()
{
    const size_t len = config.len;
    const int threads = 64;
    const int blocks  = static_cast<int>((len + threads - 1) / threads);
    rkBroadcastInitKernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(sim->device->QKv.data()) + (sim->device->QKv.size() - len),
        thrust::raw_pointer_cast(sim->device->QRv.data()) + (sim->device->QRv.size() - len),
        thrust::raw_pointer_cast(rk->device->gK.data()),
        thrust::raw_pointer_cast(rk->device->gK0.data()),
        thrust::raw_pointer_cast(rk->device->gKfinal.data()),
        thrust::raw_pointer_cast(rk->device->gKe.data()),
        thrust::raw_pointer_cast(rk->device->gR.data()),
        thrust::raw_pointer_cast(rk->device->gR0.data()),
        thrust::raw_pointer_cast(rk->device->gRfinal.data()),
        thrust::raw_pointer_cast(rk->device->gRe.data()),
        thrust::raw_pointer_cast(sim->device->error_result.data()),
        sim->device->error_result.size(),
        len);
}

// GPU Runge-Kutta initialization functions
void init_RK54GPU() {
    rk->device->init = 1;
    rk->device->stages = 7;

    rk->device->avec = new double[rk->device->stages * (rk->device->stages + 1) / 2 + 1] {1.0 / 5, 3.0 / 40, 9.0 / 40, 44.0 / 45, 
        -56.0 / 15, 32.0 / 9, 19372.0/6561, -25360.0/2187, 64448.0/6561, -212.0/729, 
        9017.0/3168, -355.0/33, 46732.0/5247, 49.0/176, -5103.0/18656, 35.0/384, 
        0.0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84};

    rk->device->bvec = new double[rk->device->stages] { 35.0/384, 0.0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84, 0.0 };

    rk->device->b2vec = new double[rk->device->stages] { 5179.0/57600, 0.0, 7571.0/16695, 393.0/640, -92097.0/339200, 187.0/2100, 1.0/40 };

    rk->device->cvec = new double[rk->device->stages] { 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1.0, 1.0};

    rk->device->d_avec.resize(rk->device->stages * (rk->device->stages + 1) / 2 + 1);
    cudaError_t err = cudaMemcpy(
        thrust::raw_pointer_cast(rk->device->d_avec.data()),
        rk->device->avec,
        rk->device->d_avec.size() * sizeof(double),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy(rk->device->d_avec) failed: ") + cudaGetErrorString(err));
    }

    rk->device->gK0.resize(config.len, 0.0);
    rk->device->gR0.resize(config.len, 0.0);
    rk->device->gK.resize(config.len, 0.0);
    rk->device->gR.resize(config.len, 0.0);
    rk->device->gKe.resize(config.len, 0.0);
    rk->device->gRe.resize(config.len, 0.0);
    rk->device->gKfinal.resize(config.len, 0.0);
    rk->device->gRfinal.resize(config.len, 0.0);

    rk->device->hK.resize(config.len * rk->device->stages, 0.0);
    rk->device->hR.resize(config.len * rk->device->stages, 0.0);
    rk->device->hK0.resize(config.len, 0.0);
    rk->device->hR0.resize(config.len, 0.0);

    if (config.debug) {
        // Basic size sanity inits
        const size_t expected = static_cast<size_t>(config.len) * static_cast<size_t>(rk->device->stages);
        if (rk->device->hK.size() != expected) {
            std::cerr << dmfe::console::ERR() << "RK54GPU init: hK size mismatch (" << rk->device->hK.size() << " vs " << expected << ")" << std::endl;
            abort();
        }
    }
}

void init_SSPRK104GPU() {
    rk->device->stages = 10;
    rk->device->posCount = 3;
    rk->device->init = 2;

    rk->device->avec = new double[rk->device->stages] {
        1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6,
        1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6
    };

    rk->device->bvec = new double[rk->device->stages] {
        1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10,
        1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10, 1.0 / 10
    };

    rk->device->b2vec = new double[rk->device->stages] {
        0.0, 2.0 / 9, 0.0, 0.0, 5.0 / 18, 1.0 / 3, 0.0, 0.0, 0.0, 1.0 / 6
    };

    rk->device->gK0.resize(config.len, 0.0);
    rk->device->gR0.resize(config.len, 0.0);
    rk->device->gK.resize(config.len, 0.0);
    rk->device->gR.resize(config.len, 0.0);
    rk->device->gKfinal.resize(config.len, 0.0);
    rk->device->gRfinal.resize(config.len, 0.0);
    rk->device->gKe.resize(config.len, 0.0);
    rk->device->gRe.resize(config.len, 0.0);

    rk->device->posB1xvec.resize(rk->device->posCount * config.len, 0.0);
    rk->device->posB2xvec.resize(rk->device->posCount * config.len * config.len, 0.0);

    rk->device->hK.resize(config.len, 0.0);
    rk->device->hR.resize(config.len, 0.0);
    rk->device->hK0.resize(config.len, 0.0);
    rk->device->hR0.resize(config.len, 0.0);

    if (config.debug) {
        const size_t expected = static_cast<size_t>(config.len);
        if (rk->device->hK.size() != expected) {
            std::cerr << dmfe::console::ERR() << "SSPRK104GPU init: hK size mismatch (" << rk->device->hK.size() << " vs " << expected << ")" << std::endl;
            abort();
        }
    }
}

// Chebyshev polynomial of the first kind T_n(x) with long double precision
long double chebyshevT_ld(int n, long double x) {
    if (n == 0) return 1.0L;
    if (n == 1) return x;
    long double t0 = 1.0L, t1 = x;
    for (int i = 2; i <= n; ++i) {
        long double t2 = 2.0L * x * t1 - t0;
        t0 = t1;
        t1 = t2;
    }
    return t1;
}

// Chebyshev polynomial of the second kind U_n(x) with long double precision
long double chebyshevU_ld(int n, long double x) {
    if (n == 0) return 1.0L;
    if (n == 1) return 2.0L * x;
    long double u0 = 1.0L, u1 = 2.0L * x;
    for (int i = 2; i <= n; ++i) {
        long double u2 = 2.0L * x * u1 - u0;
        u0 = u1;
        u1 = u2;
    }
    return u1;
}

// Gaussian elimination solver with long double precision
std::vector<long double> gaussianElimination_ld(std::vector<std::vector<long double>> A, std::vector<long double> b) {
    int n = A.size();
    std::vector<std::vector<long double>> augmented = A;
    for (int i = 0; i < n; ++i) {
        augmented[i].push_back(b[i]);
    }
    
    // Forward elimination with partial pivoting
    for (int i = 0; i < n; ++i) {
        // Find pivot
        int maxRow = i;
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(augmented[k][i]) > std::abs(augmented[maxRow][i])) {
                maxRow = k;
            }
        }
        
        // Swap rows
        std::swap(augmented[i], augmented[maxRow]);
        
        // Eliminate
        for (int k = i + 1; k < n; ++k) {
            long double factor = augmented[k][i] / augmented[i][i];
            for (int j = i; j <= n; ++j) {
                augmented[k][j] -= factor * augmented[i][j];
            }
        }
    }
    
    // Back substitution
    std::vector<long double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = augmented[i][n];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= augmented[i][j] * x[j];
        }
        x[i] /= augmented[i][i];
    }
    
    return x;
}

// Main SERKcoeffs function with enhanced precision
std::vector<double> SERKcoeffs(int q) {
    const int m = 2;
    const long double μp = 19.0L / 20.0L;
    const int s = m * q;
    const long double α2 = static_cast<long double>(q) / (static_cast<long double>(s) * s);
    const long double w0 = 1.0L + μp / (static_cast<long double>(s) * s);
    const long double w1 = chebyshevT_ld(s, w0) / (static_cast<long double>(s) * chebyshevU_ld(s - 1, w0));
    
    const int num_eq = s + 1;
    const int num_var = s + 1;
    
    std::vector<std::vector<long double>> A(num_eq, std::vector<long double>(num_var, 0.0L));
    std::vector<long double> R_values(num_eq);
    
    for (int z_idx = 0; z_idx < num_eq; ++z_idx) {
        long double z = static_cast<long double>(z_idx + 1);
        long double R_z = chebyshevT_ld(s, w0 + w1 * z) / chebyshevT_ld(s, w0);
        R_values[z_idx] = R_z;
        
        // Coefficient for b[0] (bvec[[1]] in Mathematica)
        long double T0 = chebyshevT_ld(0, 1.0L + α2 * z);
        A[z_idx][0] = T0;
        
        // Coefficients for the sum terms
        for (int j = 1; j <= q; ++j) {
            long double Tm_j = std::pow(chebyshevT_ld(m, 1.0L + α2 * z), j - 1);
            for (int i = 1; i <= m; ++i) {
                long double Ti = chebyshevT_ld(i, 1.0L + α2 * z);
                int k = i + m * (j - 1);
                A[z_idx][k] = Ti * Tm_j;
            }
        }
    }
    
    // Solve the system A * bvec = R_values using long double precision
    std::vector<long double> bvec_ld = gaussianElimination_ld(A, R_values);
    
    // Convert back to double for output
    std::vector<double> bvec(bvec_ld.size());
    for (size_t i = 0; i < bvec_ld.size(); ++i) {
        bvec[i] = static_cast<double>(bvec_ld[i]);
    }
    
    return bvec;
}

void init_SERK2(int q) {
    std::vector<double> coeffs = SERKcoeffs(q);
    rk->device->stages = coeffs.size() - 1;
    rk->device->init = 2 + q / 2;

    rk->device->bvec = new double[rk->device->stages + 1];
    for (size_t i = 0; i <= rk->device->stages; ++i) {
        rk->device->bvec[i] = coeffs[i];
    }

    rk->device->gK0.resize(config.len, 0.0);
    rk->device->gR0.resize(config.len, 0.0);
    rk->device->gK.resize(config.len, 0.0);
    rk->device->gR.resize(config.len, 0.0);
    rk->device->gKfinal.resize(config.len, 0.0);
    rk->device->gRfinal.resize(config.len, 0.0);
    rk->device->gKe.resize(config.len, 0.0);
    rk->device->gRe.resize(config.len, 0.0);

    rk->device->hK.resize(config.len, 0.0);
    rk->device->hR.resize(config.len, 0.0);
    rk->device->hK0.resize(config.len, 0.0);
    rk->device->hR0.resize(config.len, 0.0);
}

// GPU Runge-Kutta methods
double RK54GPU(StreamPool* pool) {

    size_t t1len = sim->device->t1grid.size();
    // [PERF] host mirror instead of a blocking one-element D2H copy; see
    // core/t1_cache.hpp -- the value is the very double the host wrote.
    double t_current = dmfe::t1LastGPU();
    // Initialize variables
    // [PERF] eight synchronous thrust::copy + one thrust::fill -> one kernel
    rkBroadcastInit();
    rk->device->gt = t_current;
    rk->device->gt0 = t_current;
    rk->device->gtfinal = t_current + config.delta_t;
    rk->device->gte = t_current + config.delta_t;

    double dr = 0.0;
    int threads = 64;
    int blocks = (config.len + threads - 1) / threads;

    // Loop over stages
    for (size_t n = 0; n < rk->device->stages; ++n) {
        // Interpolation
        if (sim->device->QKv.size() == config.len || n != 0) {
            // [PERF] was interpolateGPU(): with no pool argument the call fell back to
            // getDefaultStreamPool(), i.e. a *different* set of streams from the ones the
            // rest of the stage uses, which forces the two chains to be ordered only by
            // the device-wide barriers.  Passing the caller's pool is bit-exact (it only
            // changes which stream a kernel is issued on) and is what makes the event
            // based join below possible.
            interpolateGPU(nullptr, nullptr, false, pool);
        }

        QKRstepGPU(sim->device->QKv, sim->device->QRv, sim->device->QKB1int, sim->device->QKB2int, sim->device->QKA1int, sim->device->QRA1int, sim->device->QRA2int, sim->device->QRB1int, sim->device->QRB2int, sim->device->SigmaRA1int, sim->device->SigmaRA2int, sim->device->SigmaKB1int, sim->device->SigmaKB2int, sim->device->SigmaKA1int, sim->device->SigmaRB1int, sim->device->SigmaRB2int, sim->device->integ, sim->device->theta, sim->device->t1grid, sim->device->rInt, rk->device->hK, rk->device->hR, config.T0, config.Gamma, n, *pool);
        rk->device->ht = 1.0;

        // Update g and dr
        if (n == 0) {
            // [PERF] two synchronous thrust copies -> one kernel; see rkCopyH0Kernel.
            rkCopyH0Both(config.len);
            // [PERF] the two computeWeightedSum launches become one batched launch.
            {
                WSumBatchDesc wb;
                wb.add(thrust::raw_pointer_cast(rk->device->gK0.data()),
                       thrust::raw_pointer_cast(rk->device->hK.data()),
                       thrust::raw_pointer_cast(rk->device->gK.data()));
                wb.add(thrust::raw_pointer_cast(rk->device->gR0.data()),
                       thrust::raw_pointer_cast(rk->device->hR.data()),
                       thrust::raw_pointer_cast(rk->device->gR.data()));
                computeWeightedSumBatchedGPU(wb, thrust::raw_pointer_cast(rk->device->d_avec.data()), config.delta_t, n + 1, config.len, 0);
            }
            if (config.debug) DMFE_CUDA_POSTLAUNCH("RK54GPU::computeWeightedSum gK0/gR0");
            // [PERF] the up-to-four computeMA launches become one batched launch.
            // They read hK/hR and write four distinct arrays, so no entry depends on
            // another; the arithmetic of each element is unchanged.
            {
                MABatchDesc mb;
                const double* hKn = thrust::raw_pointer_cast(rk->device->hK.data()) + n * config.len;
                const double* hRn = thrust::raw_pointer_cast(rk->device->hR.data()) + n * config.len;
                if (rk->device->bvec[n] != 0.0) {
                    const double w = config.delta_t * rk->device->bvec[n];
                    mb.add(thrust::raw_pointer_cast(rk->device->gKfinal.data()), hKn, w);
                    mb.add(thrust::raw_pointer_cast(rk->device->gRfinal.data()), hRn, w);
                }
                if (rk->device->b2vec[n] != 0.0) {
                    const double w = config.delta_t * rk->device->b2vec[n];
                    mb.add(thrust::raw_pointer_cast(rk->device->gKe.data()), hKn, w);
                    mb.add(thrust::raw_pointer_cast(rk->device->gRe.data()), hRn, w);
                }
                computeMABatchedGPU(mb, config.len, 0);
            }
            if (config.debug) DMFE_CUDA_POSTLAUNCH("RK54GPU::computeMA batch");
            rk->device->gt = rk->device->gt0 + config.delta_t * rk->device->cvec[n] * rk->device->ht;
            dr = drstep2GPU(get_slice_ptr(sim->device->QKv, t1len - 1, config.len), get_slice_ptr(sim->device->QRv, t1len - 1, config.len),
                            thrust::device_pointer_cast(rk->device->hK0.data()),
                            thrust::device_pointer_cast(rk->device->hR0.data()),
                            dmfe::t1LastGPU(), config.T0, *pool);
            appendAllGPU_ptr(thrust::device_pointer_cast(rk->device->gK.data()),
                             thrust::device_pointer_cast(rk->device->gR.data()),
                             thrust::device_pointer_cast(rk->device->hK0.data()),
                             thrust::device_pointer_cast(rk->device->hR0.data()),
                             rk->device->ht * dr, rk->device->gt, config.len, *pool); // Append Update
        } else {
            // [PERF] the up-to-four computeMA launches become one batched launch.
            // They read hK/hR and write four distinct arrays, so no entry depends on
            // another; the arithmetic of each element is unchanged.
            {
                MABatchDesc mb;
                const double* hKn = thrust::raw_pointer_cast(rk->device->hK.data()) + n * config.len;
                const double* hRn = thrust::raw_pointer_cast(rk->device->hR.data()) + n * config.len;
                if (rk->device->bvec[n] != 0.0) {
                    const double w = config.delta_t * rk->device->bvec[n];
                    mb.add(thrust::raw_pointer_cast(rk->device->gKfinal.data()), hKn, w);
                    mb.add(thrust::raw_pointer_cast(rk->device->gRfinal.data()), hRn, w);
                }
                if (rk->device->b2vec[n] != 0.0) {
                    const double w = config.delta_t * rk->device->b2vec[n];
                    mb.add(thrust::raw_pointer_cast(rk->device->gKe.data()), hKn, w);
                    mb.add(thrust::raw_pointer_cast(rk->device->gRe.data()), hRn, w);
                }
                computeMABatchedGPU(mb, config.len, 0);
            }
            if (config.debug) DMFE_CUDA_POSTLAUNCH("RK54GPU::computeMA batch");
            if (n != rk->device->stages - 1) {
            // [PERF] the two computeWeightedSum launches become one batched launch.
            {
                WSumBatchDesc wb;
                wb.add(thrust::raw_pointer_cast(rk->device->gK0.data()),
                       thrust::raw_pointer_cast(rk->device->hK.data()),
                       thrust::raw_pointer_cast(rk->device->gK.data()));
                wb.add(thrust::raw_pointer_cast(rk->device->gR0.data()),
                       thrust::raw_pointer_cast(rk->device->hR.data()),
                       thrust::raw_pointer_cast(rk->device->gR.data()));
                computeWeightedSumBatchedGPU(wb, thrust::raw_pointer_cast(rk->device->d_avec.data()) + n * (n + 1) / 2, config.delta_t, n + 1, config.len, 0);
            }
            if (config.debug) DMFE_CUDA_POSTLAUNCH("RK54GPU::computeWeightedSum gK0/gR0");
                rk->device->gt = rk->device->gt0 + config.delta_t * rk->device->cvec[n] * rk->device->ht;
                replaceAllGPU_ptr(thrust::device_pointer_cast(rk->device->gK.data()),
                                  thrust::device_pointer_cast(rk->device->gR.data()),
                                  thrust::device_pointer_cast(rk->device->hK0.data()),
                                  thrust::device_pointer_cast(rk->device->hR0.data()),
                                  rk->device->ht * dr, rk->device->gt, config.len, *pool); // Replace Update
            } else {
                replaceAllGPU_ptr(thrust::device_pointer_cast(rk->device->gKfinal.data()),
                                  thrust::device_pointer_cast(rk->device->gRfinal.data()),
                                  thrust::device_pointer_cast(rk->device->hK0.data()),
                                  thrust::device_pointer_cast(rk->device->hR0.data()),
                                  rk->device->ht * dr, rk->device->gtfinal, config.len, *pool); // Replace Update
            }
        }
    }

    // Final interpolation
    interpolateGPU(thrust::raw_pointer_cast(sim->device->posB1xOld.data()), thrust::raw_pointer_cast(sim->device->posB2xOld.data()), true, pool);
 
    // Compute error estimate
    computeError<<<blocks, threads, 3 * threads * sizeof(double), (*pool)[0]>>>(   // [CPC-REV] 3 shared arrays
        thrust::raw_pointer_cast(rk->device->gKfinal.data()),
        thrust::raw_pointer_cast(rk->device->gKe.data()),
        thrust::raw_pointer_cast(rk->device->gRfinal.data()),
        thrust::raw_pointer_cast(rk->device->gRe.data()),
        thrust::raw_pointer_cast(sim->device->error_result.data()),
        config.len
    );
    if (config.debug) DMFE_CUDA_POSTLAUNCH("RK54GPU::computeError");

    // [CPC-REV] the |gtfinal - gte| term is analytically zero (sum b = sum b2 = 1)
    // and only injected rounding noise ~ulp(t); removed.  Norm now selectable.
    double error = dmfe_rk_error_from_reduction(config.len);

    return error;
}

double SERK2GPU(int q, StreamPool* pool){

    double alpha = 1.0 / (4 * q);

    int threads = 64;
    int blocks = (config.len + threads - 1) / threads;

    size_t t1len = sim->device->t1grid.size();
    // [PERF] host mirror instead of a blocking one-element D2H copy; see
    // core/t1_cache.hpp -- the value is the very double the host wrote.
    double t_current = dmfe::t1LastGPU();
    // Initialize variables
    // [PERF] eight synchronous thrust::copy + one thrust::fill -> one kernel
    rkBroadcastInit();
    rk->device->gt = t_current;
    rk->device->gt0 = t_current;
    rk->device->gtfinal = t_current + config.delta_t;
    rk->device->gte = t_current;
    computeScale<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->device->gKfinal.data()), rk->device->bvec[0], config.len);
    computeScale<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->device->gRfinal.data()), rk->device->bvec[0], config.len);

    double dr = 0.0;

    // Loop over stages
    for (size_t n = 0; n < rk->device->stages; ++n) {
        // Interpolation
        // [PERF] `pool` used to bind to the third parameter (const bool same), so the
    // StreamPool* silently became `same = true` and the pool defaulted to
    // getDefaultStreamPool().  posB1xIn/posB2xIn/same are in fact unused by
    // interpolateGPU, so only the stream pool was affected.
    interpolateGPU(thrust::raw_pointer_cast(sim->device->posB1xOld.data()), thrust::raw_pointer_cast(sim->device->posB2xOld.data()), false, pool);

        QKRstepGPU(sim->device->QKv, sim->device->QRv, sim->device->QKB1int, sim->device->QKB2int, sim->device->QKA1int, sim->device->QRA1int, sim->device->QRA2int, sim->device->QRB1int, sim->device->QRB2int, sim->device->SigmaRA1int, sim->device->SigmaRA2int, sim->device->SigmaKB1int, sim->device->SigmaKB2int, sim->device->SigmaKA1int, sim->device->SigmaRB1int, sim->device->SigmaRB2int, sim->device->integ, sim->device->theta, sim->device->t1grid, sim->device->rInt, rk->device->hK, rk->device->hR, config.T0, config.Gamma, n, *pool);
        rk->device->ht = 1.0;

        // Update g and dr
        if (n == 0) {
            // [PERF] two synchronous thrust copies -> one kernel; see rkCopyH0Kernel.
            rkCopyH0Both(config.len);
            // [PERF] two batched launches instead of four.  The pairs must stay
            // separate: the second pair reads gK/gR, which the first pair writes.
            {
                MABatchDesc mb;
                const double w = config.delta_t * alpha;
                mb.add(thrust::raw_pointer_cast(rk->device->gK.data()),
                       thrust::raw_pointer_cast(rk->device->hK.data()) + n * config.len, w);
                mb.add(thrust::raw_pointer_cast(rk->device->gR.data()),
                       thrust::raw_pointer_cast(rk->device->hR.data()) + n * config.len, w);
                computeMABatchedGPU(mb, config.len, 0);
            }
            if (config.debug) DMFE_CUDA_POSTLAUNCH("SERK2GPU::computeMA gK/gR");
            if(rk->device->bvec[n + 1] != 0.0) {
                MABatchDesc mb;
                const double w = rk->device->bvec[n + 1];
                mb.add(thrust::raw_pointer_cast(rk->device->gKfinal.data()),
                       thrust::raw_pointer_cast(rk->device->gK.data()), w);
                mb.add(thrust::raw_pointer_cast(rk->device->gRfinal.data()),
                       thrust::raw_pointer_cast(rk->device->gR.data()), w);
                computeMABatchedGPU(mb, config.len, 0);
                if (config.debug) DMFE_CUDA_POSTLAUNCH("SERK2GPU::computeMA gKfinal/gRfinal");
            }
            rk->device->gt = rk->device->gt0 + config.delta_t * alpha * rk->device->ht;
            dr = drstep2GPU(get_slice_ptr(sim->device->QKv, t1len - 1, config.len), get_slice_ptr(sim->device->QRv, t1len - 1, config.len),
                            thrust::device_pointer_cast(rk->device->hK0.data()),
                            thrust::device_pointer_cast(rk->device->hR0.data()),
                            dmfe::t1LastGPU(), config.T0, *pool);
            appendAllGPU_ptr(thrust::device_pointer_cast(rk->device->gK.data()),
                             thrust::device_pointer_cast(rk->device->gR.data()),
                             thrust::device_pointer_cast(rk->device->hK0.data()),
                             thrust::device_pointer_cast(rk->device->hR0.data()),
                             rk->device->ht * dr, rk->device->gt, config.len, *pool); // Append Update
        } else {
            if (n % 2 == 0){
                // [PERF] one batched launch instead of two; see MABatchDesc.
                MABatchDesc mb;
                const double w = config.delta_t * alpha;
                mb.add(thrust::raw_pointer_cast(rk->device->gK.data()),
                       thrust::raw_pointer_cast(rk->device->hK.data()) + n * config.len, w);
                mb.add(thrust::raw_pointer_cast(rk->device->gR.data()),
                       thrust::raw_pointer_cast(rk->device->hR.data()) + n * config.len, w);
                computeMABatchedGPU(mb, config.len, 0);
                rk->device->gt +=  config.delta_t * alpha * rk->device->ht;
            } else {
                computeMAD<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->device->gK.data()), thrust::raw_pointer_cast(rk->device->gKe.data()), thrust::raw_pointer_cast(rk->device->hK.data()) + n * config.len, 2 * config.delta_t * alpha, config.len);
                computeMAD<<<blocks, threads>>>(thrust::raw_pointer_cast(rk->device->gR.data()), thrust::raw_pointer_cast(rk->device->gRe.data()), thrust::raw_pointer_cast(rk->device->hR.data()) + n * config.len, 2 * config.delta_t * alpha, config.len);
                rk->device->gt = 2 * rk->device->gt - rk->device->gte + 2 * config.delta_t * alpha * rk->device->ht;
                // [PERF] two synchronous thrust copies -> one kernel (pure data
                // movement, legacy default stream, as before).
                rkCopy2Both(thrust::raw_pointer_cast(rk->device->gK.data()),
                            thrust::raw_pointer_cast(rk->device->gR.data()),
                            thrust::raw_pointer_cast(rk->device->gKe.data()),
                            thrust::raw_pointer_cast(rk->device->gRe.data()),
                            config.len);
                rk->device->gte = rk->device->gt;
            }
            if(rk->device->bvec[n + 1] != 0.0) {
                // [PERF] one batched launch instead of two; see MABatchDesc.
                MABatchDesc mb;
                const double w = rk->device->bvec[n + 1];
                mb.add(thrust::raw_pointer_cast(rk->device->gKfinal.data()),
                       thrust::raw_pointer_cast(rk->device->gK.data()), w);
                mb.add(thrust::raw_pointer_cast(rk->device->gRfinal.data()),
                       thrust::raw_pointer_cast(rk->device->gR.data()), w);
                computeMABatchedGPU(mb, config.len, 0);
            }
            replaceAllGPU_ptr(thrust::device_pointer_cast(rk->device->gK.data()),
                              thrust::device_pointer_cast(rk->device->gR.data()),
                              thrust::device_pointer_cast(rk->device->hK0.data()),
                              thrust::device_pointer_cast(rk->device->hR0.data()),
                              rk->device->ht * dr, rk->device->gt, config.len, *pool); // Replace Update
        }
    }
 
    // Compute error estimate
    computeError<<<blocks, threads, 3 * threads * sizeof(double), (*pool)[0]>>>(   // [CPC-REV] 3 shared arrays
        thrust::raw_pointer_cast(rk->device->gKfinal.data()),
        thrust::raw_pointer_cast(rk->device->gK.data()),
        thrust::raw_pointer_cast(rk->device->gRfinal.data()),
        thrust::raw_pointer_cast(rk->device->gR.data()),
        thrust::raw_pointer_cast(sim->device->error_result.data()),
        config.len
    );
    if (config.debug) DMFE_CUDA_POSTLAUNCH("SERK2GPU::computeError");

    // [CPC-REV] see above: the time component is analytically zero for SERK2 as well
    // (the Chebyshev recursion gives gt - t = Delta t exactly), but on the host it is
    // accumulated in floating point and reaches ~2e-8 at t = 1e7.  Removed.
    double error = dmfe_rk_error_from_reduction(config.len);

    return error;
}

double SSPRK104GPU(StreamPool* pool) {

    size_t t1len = sim->device->t1grid.size();
    // [PERF] host mirror instead of a blocking one-element D2H copy; see
    // core/t1_cache.hpp -- the value is the very double the host wrote.
    double t_current = dmfe::t1LastGPU();
    // Initialize variables
    // [PERF] eight synchronous thrust::copy + one thrust::fill -> one kernel
    rkBroadcastInit();
    rk->device->gt = t_current;
    rk->device->gt0 = t_current;
    rk->device->gtfinal = t_current + config.delta_t;
    rk->device->gte = t_current + config.delta_t;
    
    double dr = 0.0;
    int threads = 64;
    int blocks = (config.len + threads - 1) / threads;

    // Loop over stages
    for (size_t n = 0; n < rk->device->stages; ++n) {
        // Interpolation
        if (sim->device->QKv.size() == config.len || n != 0) {
            interpolateGPU(
                (n == 0 ? nullptr : (n == 5 ? get_slice_ptr(rk->device->posB1xvec, 0, config.len).get() : (n == 6 ? get_slice_ptr(rk->device->posB1xvec, 1, config.len).get() : (n == 7 ? get_slice_ptr(rk->device->posB1xvec, 2, config.len).get() : thrust::raw_pointer_cast(sim->device->posB1xOld.data()))))),
                (n == 0 ? nullptr : (n == 5 ? get_slice_ptr(rk->device->posB2xvec, 0, config.len*config.len).get() : (n == 6 ? get_slice_ptr(rk->device->posB2xvec, 1, config.len*config.len).get() : (n == 7 ? get_slice_ptr(rk->device->posB2xvec, 2, config.len*config.len).get() : thrust::raw_pointer_cast(sim->device->posB2xOld.data()))))),
                (n == 5 || n == 6 || n == 7),
                pool);
        }

        // Update position vectors
        if (n == 2) {
            set_slice(rk->device->posB1xvec,0,sim->device->posB1xOld);
            set_slice(rk->device->posB2xvec,0,sim->device->posB2xOld);
        } else if (n == 3) {
            set_slice(rk->device->posB1xvec,1,sim->device->posB1xOld);
            set_slice(rk->device->posB2xvec,1,sim->device->posB2xOld);
        } else if (n == 4) {
            set_slice(rk->device->posB1xvec,2,sim->device->posB1xOld);
            set_slice(rk->device->posB2xvec,2,sim->device->posB2xOld);
        }

        QKRstepGPU(sim->device->QKv, sim->device->QRv, sim->device->QKB1int, sim->device->QKB2int, sim->device->QKA1int, sim->device->QRA1int, sim->device->QRA2int, sim->device->QRB1int, sim->device->QRB2int, sim->device->SigmaRA1int, sim->device->SigmaRA2int, sim->device->SigmaKB1int, sim->device->SigmaKB2int, sim->device->SigmaKA1int, sim->device->SigmaRB1int, sim->device->SigmaRB2int, sim->device->integ, sim->device->theta, sim->device->t1grid, sim->device->rInt, rk->device->hK, rk->device->hR, config.T0, config.Gamma, 0, *pool);
        rk->device->ht = 1.0;


        // Update g and dr
        if (n == 0) {
            // [PERF] two synchronous thrust copies -> one kernel; see rkCopyH0Kernel.
            // The resize keeps the sizes that operator= used to establish, and the
            // kernel then moves exactly the same elements.
            if (rk->device->hK0.size() != rk->device->hK.size()) rk->device->hK0.resize(rk->device->hK.size());
            if (rk->device->hR0.size() != rk->device->hR.size()) rk->device->hR0.resize(rk->device->hR.size());
            rkCopyH0Both(rk->device->hK.size());
            // [PERF] the four-to-six computeMA launches become one batched launch.
            // They all read hK/hR and write distinct arrays (gK, gR, gKfinal, gRfinal,
            // gKe, gRe), so no entry depends on another; each element is computed by
            // the same expression as before.
            {
                MABatchDesc mb;
                const double* hK0p = thrust::raw_pointer_cast(rk->device->hK.data());
                const double* hR0p = thrust::raw_pointer_cast(rk->device->hR.data());
                const double wa = config.delta_t * rk->device->avec[n];
                const double wb = config.delta_t * rk->device->bvec[n];
                mb.add(thrust::raw_pointer_cast(rk->device->gK.data()),      hK0p, wa);
                mb.add(thrust::raw_pointer_cast(rk->device->gR.data()),      hR0p, wa);
                mb.add(thrust::raw_pointer_cast(rk->device->gKfinal.data()), hK0p, wb);
                mb.add(thrust::raw_pointer_cast(rk->device->gRfinal.data()), hR0p, wb);
                if (rk->device->b2vec[n] != 0.0) {
                    const double w2 = config.delta_t * rk->device->b2vec[n];
                    mb.add(thrust::raw_pointer_cast(rk->device->gKe.data()), hK0p, w2);
                    mb.add(thrust::raw_pointer_cast(rk->device->gRe.data()), hR0p, w2);
                }
                computeMABatchedGPU(mb, config.len, 0);
            }
            if (config.debug) DMFE_CUDA_POSTLAUNCH("SSPRK104GPU::computeMA batch");
            rk->device->gt += config.delta_t * rk->device->avec[n] * rk->device->ht;
            dr = drstep2GPU(get_slice_ptr(sim->device->QKv, t1len - 1, config.len), get_slice_ptr(sim->device->QRv, t1len - 1, config.len),
                            thrust::device_pointer_cast(rk->device->hK.data()),
                            thrust::device_pointer_cast(rk->device->hR.data()),
                            t_current, config.T0, *pool);

            appendAllGPU_ptr(thrust::device_pointer_cast(rk->device->gK.data()),
                             thrust::device_pointer_cast(rk->device->gR.data()),
                             thrust::device_pointer_cast(rk->device->hK.data()),
                             thrust::device_pointer_cast(rk->device->hR.data()),
                             rk->device->ht * dr, rk->device->gt, config.len, *pool); // Append Update
        } else {
            // [PERF] see the n == 0 branch: one batched launch instead of up to six.
            {
                MABatchDesc mb;
                const double* hK0p = thrust::raw_pointer_cast(rk->device->hK.data());
                const double* hR0p = thrust::raw_pointer_cast(rk->device->hR.data());
                if (n != rk->device->stages - 1) {
                    const double wa = config.delta_t * rk->device->avec[n];
                    mb.add(thrust::raw_pointer_cast(rk->device->gK.data()), hK0p, wa);
                    mb.add(thrust::raw_pointer_cast(rk->device->gR.data()), hR0p, wa);
                }
                const double wb = config.delta_t * rk->device->bvec[n];
                mb.add(thrust::raw_pointer_cast(rk->device->gKfinal.data()), hK0p, wb);
                mb.add(thrust::raw_pointer_cast(rk->device->gRfinal.data()), hR0p, wb);
                if (rk->device->b2vec[n] != 0.0) {
                    const double w2 = config.delta_t * rk->device->b2vec[n];
                    mb.add(thrust::raw_pointer_cast(rk->device->gKe.data()), hK0p, w2);
                    mb.add(thrust::raw_pointer_cast(rk->device->gRe.data()), hR0p, w2);
                }
                computeMABatchedGPU(mb, config.len, 0);
            }
            if (n == 4) {
                AddSubtractGPU(rk->device->gK, rk->device->gKfinal, rk->device->gK0, rk->device->gR, rk->device->gRfinal, rk->device->gR0, (*pool)[0]);
                rk->device->gt = rk->device->gt0*9/15 + (rk->device->gt + config.delta_t * rk->device->avec[n] * rk->device->ht)*6/15;
            } else if (n != rk->device->stages - 1) {
                rk->device->gt += config.delta_t * rk->device->avec[n] * rk->device->ht;
            }
            if (n != rk->device->stages - 1) {
                replaceAllGPU_ptr(thrust::device_pointer_cast(rk->device->gK.data()),
                                  thrust::device_pointer_cast(rk->device->gR.data()),
                                  thrust::device_pointer_cast(rk->device->hK0.data()),
                                  thrust::device_pointer_cast(rk->device->hR0.data()),
                                  rk->device->ht * dr, rk->device->gt, config.len, *pool); // Replace Update
            } else {
                replaceAllGPU_ptr(thrust::device_pointer_cast(rk->device->gKfinal.data()),
                                  thrust::device_pointer_cast(rk->device->gRfinal.data()),
                                  thrust::device_pointer_cast(rk->device->hK0.data()),
                                  thrust::device_pointer_cast(rk->device->hR0.data()),
                                  rk->device->ht * dr, rk->device->gt, config.len, *pool); // Replace Update
            }
        }
    }

    // Final interpolation
    // [PERF] `pool` used to bind to the third parameter (const bool same), so the
    // StreamPool* silently became `same = true` and the pool defaulted to
    // getDefaultStreamPool().  posB1xIn/posB2xIn/same are in fact unused by
    // interpolateGPU, so only the stream pool was affected.
    interpolateGPU(thrust::raw_pointer_cast(sim->device->posB1xOld.data()), thrust::raw_pointer_cast(sim->device->posB2xOld.data()), false, pool);

    // Compute error estimate
    computeError<<<blocks, threads, 3 * threads * sizeof(double), (*pool)[0]>>>(   // [CPC-REV] 3 shared arrays
        thrust::raw_pointer_cast(rk->device->gKfinal.data()),
        thrust::raw_pointer_cast(rk->device->gKe.data()),
        thrust::raw_pointer_cast(rk->device->gRfinal.data()),
        thrust::raw_pointer_cast(rk->device->gRe.data()),
        thrust::raw_pointer_cast(sim->device->error_result.data()),
        config.len
    );
    if (config.debug) DMFE_CUDA_POSTLAUNCH("SSPRK104GPU::computeError");

    // [CPC-REV] the |gtfinal - gte| term is analytically zero (sum b = sum b2 = 1)
    // and only injected rounding noise ~ulp(t); removed.  Norm now selectable.
    double error = dmfe_rk_error_from_reduction(config.len);

    return error;
}

// GPU method selection function
double updateGPU(StreamPool* pool) {
    if (rk->device->init == 1) {
        return RK54GPU(pool);
    } else if (rk->device->init == 2) {
        return SSPRK104GPU(pool);
    } else {
        return SERK2GPU(2 * (rk->device->init - 2), pool);
    }
}
