#include "interpolation/index_vec.cuh"
#include "core/globals.hpp"
         // for sim global pointer
#include "math/math_ops.hpp"        // for pow_const template
#include <vector>
#include <numeric>             // for std::inner_product
#include <thrust/copy.h>
#include <thrust/transform.h>

// ================= indexVecLN3 =====================

// ============================================================================
// [PERF] NOTE on the weights layout (measured, do not "fix" without re-measuring).
//
// The stencil weights arrive from Grid_data in j-major order: the `depth` weights
// of output j are contiguous, so a warp reads, for each d, 32 addresses 8*depth
// bytes apart.  That looks like a textbook uncoalesced access, and indexVecLN3 +
// indexMatAll are ~48% of all GPU time, so it is the obvious thing to attack.  It
// is a trap: the ~20 cache lines a warp touches for one d are exactly the lines
// the other depth-1 loads need, so the DRAM traffic is already minimal and only
// the L1 request count is high.  Three bit-exact rewrites were tried on an H100
// and all three are SLOWER:
//   * d-major (transposed) device copies, fully coalesced:  42.4 -> 46.8 us
//     (indexVecLN3) and 32.9 -> 33.7 us (indexMatAll), plus 80 ms of host-side
//     transpose at start-up -- coalescing buys nothing because the traffic was
//     never the problem, and the per-thread stencil is now spread over `depth`
//     streams 2 MB apart;
//   * staging the block's slice in shared memory:  whole-run wall 2.44 -> 2.68 s;
//   * __launch_bounds__(64, 16) to double occupancy: 42.4 -> 46.6 us and
//     32.9 -> 45.2 us -- these kernels live on the memory-level parallelism of a
//     single thread, not on the number of resident warps.
// ============================================================================


template <int DEPTH>
__global__ __launch_bounds__(64, 1)
void indexVecLN3Kernel_specialized(
    const double* __restrict__ QKv,
    const double* __restrict__ QRv,
    const double* __restrict__ weights,
    const size_t* __restrict__ inds,
    double* __restrict__ qk_out,
    double* __restrict__ qr_out,
    size_t offset,
    size_t prod)
{
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= prod) return;

    size_t index = inds[j];
    const double* w = weights + j * DEPTH;

    double qk = 0.0, qr = 0.0;
    #pragma unroll
    for (int d = 0; d < DEPTH; ++d) {
        double wj = w[d];
        qk += wj * QKv[offset + index + d];
        qr += wj * QRv[offset + index + d];
    }

    qk_out[j] = qk;
    qr_out[j] = qr;
}

__global__ __launch_bounds__(64, 1)
void indexVecLN3Kernel_dynamic(
    const double* __restrict__ QKv,
    const double* __restrict__ QRv,
    const double* __restrict__ weights,
    const size_t* __restrict__ inds,
    double* __restrict__ qk_out,
    double* __restrict__ qr_out,
    size_t depth,
    size_t offset,
    size_t prod)
{
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= prod) return;
    size_t index = inds[j];
    const double* w = weights + j * depth;
    double qk = 0.0, qr = 0.0;
    for (int d = 0; d < depth; ++d) {
        double wj = w[d];
        qk += wj * QKv[offset + index + d];
        qr += wj * QRv[offset + index + d];
    }
    qk_out[j] = qk;
    qr_out[j] = qr;
}

// File-scope static state to avoid dependency on global SimulationData
namespace {
    const double* g_QKv_device_ptr = nullptr;
    const double* g_QRv_device_ptr = nullptr;
    size_t g_QKv_global_size = 0;
    size_t g_current_len = 0; // set each launch; used to compute offset
}

class IndexVecLN3Optimizer {
private:
    static size_t cached_depth;
    static std::function<void(const dmfe::device_vector<double>&,
                             const dmfe::device_vector<size_t>&,
                             dmfe::device_vector<double>&,
                             dmfe::device_vector<double>&,
                             cudaStream_t)> cached_kernel;

    template<int DEPTH>
    static void call_specialized(
        const dmfe::device_vector<double>& weights,
        const dmfe::device_vector<size_t>& inds,
        dmfe::device_vector<double>& qk_result,
        dmfe::device_vector<double>& qr_result,
        cudaStream_t stream)
    {
    const size_t prod = inds.size();
    const size_t offset = g_QKv_global_size - g_current_len;
    const double* QKv_ptr = g_QKv_device_ptr;
    const double* QRv_ptr = g_QRv_device_ptr;
        const double* W   = thrust::raw_pointer_cast(weights.data());
        const size_t* I   = thrust::raw_pointer_cast(inds.data());
        double* QK_out    = thrust::raw_pointer_cast(qk_result.data());
        double* QR_out    = thrust::raw_pointer_cast(qr_result.data());
        const int threads = 64;
        const int blocks = (prod + threads - 1) / threads;
        indexVecLN3Kernel_specialized<DEPTH><<<blocks, threads, 0, stream>>>(
            QKv_ptr, QRv_ptr, W, I, QK_out, QR_out, offset, prod
        );
    }

    static void call_dynamic(
        const dmfe::device_vector<double>& weights,
        const dmfe::device_vector<size_t>& inds,
        dmfe::device_vector<double>& qk_result,
        dmfe::device_vector<double>& qr_result,
        cudaStream_t stream)
    {
    const size_t prod = inds.size();
    const size_t depth = weights.size() / prod;
    const size_t offset = g_QKv_global_size - g_current_len;
    const double* QKv_ptr = g_QKv_device_ptr;
    const double* QRv_ptr = g_QRv_device_ptr;
        const double* W   = thrust::raw_pointer_cast(weights.data());
        const size_t* I   = thrust::raw_pointer_cast(inds.data());
        double* QK_out    = thrust::raw_pointer_cast(qk_result.data());
        double* QR_out    = thrust::raw_pointer_cast(qr_result.data());
        const int threads = 64;
        const int blocks = (prod + threads - 1) / threads;
        indexVecLN3Kernel_dynamic<<<blocks, threads, 0, stream>>>(
            QKv_ptr, QRv_ptr, W, I, QK_out, QR_out, depth, offset, prod
        );
    }
public:
    static void setupKernel(size_t depth) {
        if (cached_depth != depth) {
            cached_depth = depth;
            switch(depth) {
                // [PERF] NOTE: this switches on the stencil DEPTH = interpolation order + 1, not
    // on the order, so the shipped order-9 grids (depth 10) fall through to
    // call_dynamic.  We measured the unrolled specialization for depth 10 and it is
    // *slower* on an H100 (26.5 us vs 22.9 us for indexVecLN3): with
    // __launch_bounds__(64,1) the fully unrolled body raises register pressure
    // without buying latency hiding.  The even depths are therefore deliberately
    // NOT listed; re-measure before adding them.
                case 9: cached_kernel = call_specialized<9>; break;
                case 11: cached_kernel = call_specialized<11>; break;
                case 13: cached_kernel = call_specialized<13>; break;
                case 15: cached_kernel = call_specialized<15>; break;
                case 17: cached_kernel = call_specialized<17>; break;
                default: cached_kernel = call_dynamic; break;
            }
        }
    }
    static void execute(
        const dmfe::device_vector<double>& weights,
        const dmfe::device_vector<size_t>& inds,
        dmfe::device_vector<double>& qk_result,
        dmfe::device_vector<double>& qr_result,
        cudaStream_t stream)
    { cached_kernel(weights, inds, qk_result, qr_result, stream); }
};

size_t IndexVecLN3Optimizer::cached_depth = 0;
std::function<void(const dmfe::device_vector<double>&,
                  const dmfe::device_vector<size_t>&,
                  dmfe::device_vector<double>&,
                  dmfe::device_vector<double>&,
                  cudaStream_t)> IndexVecLN3Optimizer::cached_kernel;

void indexVecLN3GPU(
    const dmfe::device_vector<double>& weights,
    const dmfe::device_vector<size_t>& inds,
    const dmfe::device_vector<double>& QKv,
    const dmfe::device_vector<double>& QRv,
    size_t len,
    dmfe::device_vector<double>& qk_result,
    dmfe::device_vector<double>& qr_result,
    cudaStream_t stream)
{
    const size_t depth = weights.size() / inds.size();
    IndexVecLN3Optimizer::setupKernel(depth);
    // Provide raw pointers and total size to static members via thread-local statics
    g_QKv_device_ptr = thrust::raw_pointer_cast(QKv.data());
    g_QRv_device_ptr = thrust::raw_pointer_cast(QRv.data());
    g_QKv_global_size = QKv.size();
    g_current_len = len; // store len for offset computation inside optimizer
    IndexVecLN3Optimizer::execute(weights, inds, qk_result, qr_result, stream);
}

// ============================================================================
// [PERF] Batched LN3 interpolation.
//
// interpolateGPU() interpolates the A1 and A2 slices with two separate launches
// of this kernel per Runge-Kutta stage, on two different streams.  It is the most
// expensive kernel in the run (348 ms of 1.45 s of GPU time in a 1124-step
// benchmark) and each launch is only (prod/64) blocks, so merging the two into a
// single (blocks x 2) grid both halves the launch count and doubles the number of
// resident blocks.
//
// BIT-EXACTNESS: the body is the dynamic kernel above, character for character.
// Element j of slice k accumulates the same `depth` products in the same order
// from the same operands; there is no reduction across threads and no shared
// state.  Only the block-to-launch mapping changes.
//
// The batched path is taken only when the two slices agree in `prod` and `depth`
// and the depth is not one of the values for which a fully unrolled kernel is
// selected above; otherwise the two original launches are issued unchanged.
// ============================================================================
// [PERF] The occupancy hint was `__launch_bounds__(64, 1)`, i.e. "at least one
// 64-thread block per SM", which lets ptxas spend as many registers as it likes:
// it used 122-126 of them, so only ~8 blocks (512 of the 2048 threads an H100 SM
// can hold) were resident and there was little to hide the memory latency with.
// Asking for 16 blocks per SM does cap the budget (61 and 64 registers, no
// spills) and does double the resident warps -- and it is SLOWER: measured on an
// H100, indexVecLN3 goes 42.4 -> 46.6 us and indexMatAll 32.9 -> 45.2 us per
// launch.  These kernels are limited by the memory-level parallelism of a single
// thread (each one issues `depth` independent gathers), not by the number of
// resident warps, so the extra registers are worth more than the occupancy.  The
// hint is therefore left as it was; re-measure before changing it.
//
// (An occupancy hint is bit-exact either way: it changes register allocation and
// instruction scheduling only.  nvcc does not reassociate floating-point
// expressions, and --use_fast_math does not enable reassociation, so each thread
// evaluates the same accumulation in the same order on the same operands.)
__global__ __launch_bounds__(64, 1)
void indexVecLN3Kernel_batched(
    const double* __restrict__ QKv,
    const double* __restrict__ QRv,
    LN3PairDesc batch,
    size_t depth,
    size_t offset,
    size_t prod)
{
    // [PERF] NOTE: the stencil weights are j-major (thread j reads w[j*depth + d]),
    // so each of the `depth` loads of a warp touches ~20 cache lines for 256 useful
    // bytes.  We tried staging the block's slice in shared memory with coalesced
    // loads: it is bit-exact but *slower* on an H100 (whole-run wall 2.68 s vs
    // 2.44 s) -- L1 already absorbs the reuse, and the extra __syncthreads and
    // shared-memory footprint cost more than the coalescing buys.  Re-measure
    // before trying it again.
    const int k = blockIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= prod) return;
    size_t index = batch.inds[k][j];
    const double* w = batch.weights[k] + j * depth;
    double qk = 0.0, qr = 0.0;
    for (int d = 0; d < depth; ++d) {
        double wj = w[d];
        qk += wj * QKv[offset + index + d];
        qr += wj * QRv[offset + index + d];
    }
    batch.qk_out[k][j] = qk;
    batch.qr_out[k][j] = qr;
}

static bool dmfe_ln3_depth_is_specialized(size_t depth)
{
    switch (depth) {
        case 9: case 11: case 13: case 15: case 17: return true;
        default: return false;
    }
}

void indexVecLN3GPU_pair(
    const dmfe::device_vector<double>& weightsA,
    const dmfe::device_vector<size_t>& indsA,
    dmfe::device_vector<double>& qkA,
    dmfe::device_vector<double>& qrA,
    cudaStream_t streamA,
    const dmfe::device_vector<double>& weightsB,
    const dmfe::device_vector<size_t>& indsB,
    dmfe::device_vector<double>& qkB,
    dmfe::device_vector<double>& qrB,
    cudaStream_t streamB,
    const dmfe::device_vector<double>& QKv,
    const dmfe::device_vector<double>& QRv,
    size_t len,
    cudaStream_t batchStream)
{
    const size_t prodA  = indsA.size();
    const size_t prodB  = indsB.size();
    const size_t depthA = weightsA.size() / prodA;
    const size_t depthB = weightsB.size() / prodB;

    if (prodA != prodB || depthA != depthB || dmfe_ln3_depth_is_specialized(depthA)) {
        // Shapes differ, or an unrolled kernel is the better choice: keep the two
        // original launches, on their original streams.
        indexVecLN3GPU(weightsA, indsA, QKv, QRv, len, qkA, qrA, streamA);
        indexVecLN3GPU(weightsB, indsB, QKv, QRv, len, qkB, qrB, streamB);
        return;
    }

    LN3PairDesc b;
    b.weights[0] = thrust::raw_pointer_cast(weightsA.data());
    b.inds[0]    = thrust::raw_pointer_cast(indsA.data());
    b.qk_out[0]  = thrust::raw_pointer_cast(qkA.data());
    b.qr_out[0]  = thrust::raw_pointer_cast(qrA.data());
    b.weights[1] = thrust::raw_pointer_cast(weightsB.data());
    b.inds[1]    = thrust::raw_pointer_cast(indsB.data());
    b.qk_out[1]  = thrust::raw_pointer_cast(qkB.data());
    b.qr_out[1]  = thrust::raw_pointer_cast(qrB.data());

    const size_t offset = QKv.size() - len;
    const int threads = 64;                      // must match the single-slice kernel
    const int blocks  = static_cast<int>((prodA + threads - 1) / threads);
    dim3 grid(static_cast<unsigned>(blocks), 2u);
    indexVecLN3Kernel_batched<<<grid, threads, 0, batchStream>>>(
        thrust::raw_pointer_cast(QKv.data()),
        thrust::raw_pointer_cast(QRv.data()),
        b, depthA, offset, prodA);
}

// ================= indexVecLN3 (log QR) =====================

// Specialized kernel: computes qK in linear domain; qR in log domain when all sampled QR>0, else linear fallback.
template <int DEPTH>
__global__ __launch_bounds__(64, 1)
void indexVecLN3Kernel_specialized_log(
    const double* __restrict__ QKv_last,
    const double* __restrict__ QR_last,
    const double* __restrict__ logQR_last,
    const double* __restrict__ weights,
    const size_t* __restrict__ inds,
    double* __restrict__ qk_out,
    double* __restrict__ qr_out,
    size_t prod)
{
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= prod) return;

    size_t index = inds[j];
    const double* w = weights + j * DEPTH;

    double qk_lin = 0.0;
    double qr_log_sum = 0.0;
    bool okR = true;
    #pragma unroll
    for (int d = 0; d < DEPTH; ++d) {
        double wj = w[d];
        double qkv = QKv_last[index + d];
        double qrv = QR_last[index + d];
        qk_lin += wj * qkv;
        okR = okR && (qrv > 0.0);
        qr_log_sum += wj * logQR_last[index + d];
    }
    double qr;
    if (okR) {
        qr = exp(qr_log_sum);
    } else {
        // Fallback: linear-domain weighted sum
        qr = 0.0;
        #pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            double wj = w[d];
            qr += wj * QR_last[index + d];
        }
    }
    qk_out[j] = qk_lin;
    qr_out[j] = qr;
}

__global__ __launch_bounds__(64, 1)
void indexVecLN3Kernel_dynamic_log(
    const double* __restrict__ QKv_last,
    const double* __restrict__ QR_last,
    const double* __restrict__ logQR_last,
    const double* __restrict__ weights,
    const size_t* __restrict__ inds,
    double* __restrict__ qk_out,
    double* __restrict__ qr_out,
    size_t depth,
    size_t prod)
{
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= prod) return;
    size_t index = inds[j];
    const double* w = weights + j * depth;
    double qk_lin = 0.0;
    double qr_log_sum = 0.0;
    bool okR = true;
    for (int d = 0; d < depth; ++d) {
        double wj = w[d];
        double qkv = QKv_last[index + d];
        double qrv = QR_last[index + d];
        qk_lin += wj * qkv;
        okR = okR && (qrv > 0.0);
        qr_log_sum += wj * logQR_last[index + d];
    }
    double qr;
    if (okR) {
        qr = exp(qr_log_sum);
    } else {
        // Fallback: linear-domain weighted sum
        qr = 0.0;
        for (int d = 0; d < depth; ++d) {
            double wj = w[d];
            qr += wj * QR_last[index + d];
        }
    }
    qk_out[j] = qk_lin;
    qr_out[j] = qr;
}

class IndexVecLN3LogOptimizer {
private:
    static size_t cached_depth;
    static std::function<void(const dmfe::device_vector<double>&,
                             const dmfe::device_vector<size_t>&,
                             const double*,
                             const double*,
                             const dmfe::device_vector<double>&,
                             dmfe::device_vector<double>&,
                             dmfe::device_vector<double>&,
                             cudaStream_t)> cached_kernel;

    template<int DEPTH>
    static void call_specialized(
        const dmfe::device_vector<double>& weights,
        const dmfe::device_vector<size_t>& inds,
        const double* QKv_last_ptr,
        const double* QRv_last_ptr,
        const dmfe::device_vector<double>& logQR_last,
        dmfe::device_vector<double>& qk_result,
        dmfe::device_vector<double>& qr_result,
        cudaStream_t stream)
    {
        const size_t prod = inds.size();
        const double* W   = thrust::raw_pointer_cast(weights.data());
        const size_t* I   = thrust::raw_pointer_cast(inds.data());
        const double* logQR_ptr = thrust::raw_pointer_cast(logQR_last.data());
        double* QK_out    = thrust::raw_pointer_cast(qk_result.data());
        double* QR_out    = thrust::raw_pointer_cast(qr_result.data());
        const int threads = 64;
        const int blocks = (prod + threads - 1) / threads;
        indexVecLN3Kernel_specialized_log<DEPTH><<<blocks, threads, 0, stream>>>(
            QKv_last_ptr, QRv_last_ptr, logQR_ptr, W, I, QK_out, QR_out, prod
        );
    }

    static void call_dynamic(
        const dmfe::device_vector<double>& weights,
        const dmfe::device_vector<size_t>& inds,
        const double* QKv_last_ptr,
        const double* QRv_last_ptr,
        const dmfe::device_vector<double>& logQR_last,
        dmfe::device_vector<double>& qk_result,
        dmfe::device_vector<double>& qr_result,
        cudaStream_t stream)
    {
        const size_t prod = inds.size();
        const size_t depth = weights.size() / prod;
        const double* W   = thrust::raw_pointer_cast(weights.data());
        const size_t* I   = thrust::raw_pointer_cast(inds.data());
        const double* logQR_ptr = thrust::raw_pointer_cast(logQR_last.data());
        double* QK_out    = thrust::raw_pointer_cast(qk_result.data());
        double* QR_out    = thrust::raw_pointer_cast(qr_result.data());
        const int threads = 64;
        const int blocks = (prod + threads - 1) / threads;
        indexVecLN3Kernel_dynamic_log<<<blocks, threads, 0, stream>>>(
            QKv_last_ptr, QRv_last_ptr, logQR_ptr, W, I, QK_out, QR_out, depth, prod
        );
    }
public:
    static void setupKernel(size_t depth) {
        if (cached_depth != depth) {
            cached_depth = depth;
            switch(depth) {
                case 9:  cached_kernel = call_specialized<9>; break;
                case 11: cached_kernel = call_specialized<11>; break;
                case 13: cached_kernel = call_specialized<13>; break;
                case 15: cached_kernel = call_specialized<15>; break;
                case 17: cached_kernel = call_specialized<17>; break;
                default: cached_kernel = call_dynamic; break;
            }
        }
    }
    static void execute(
        const dmfe::device_vector<double>& weights,
        const dmfe::device_vector<size_t>& inds,
        const double* QKv_last_ptr,
        const double* QRv_last_ptr,
        const dmfe::device_vector<double>& logQR_last,
        dmfe::device_vector<double>& qk_result,
        dmfe::device_vector<double>& qr_result,
        cudaStream_t stream)
    { cached_kernel(weights, inds, QKv_last_ptr, QRv_last_ptr, logQR_last, qk_result, qr_result, stream); }
};

size_t IndexVecLN3LogOptimizer::cached_depth = 0;
std::function<void(const dmfe::device_vector<double>&,
                  const dmfe::device_vector<size_t>&,
                  const double*,
                  const double*,
                  const dmfe::device_vector<double>&,
                  dmfe::device_vector<double>&,
                  dmfe::device_vector<double>&,
                  cudaStream_t)> IndexVecLN3LogOptimizer::cached_kernel;

// Host launcher for log-space LN3
// Precompute logs of last len entries into provided output vector
void prepareLN3LogSliceGPU_into(size_t len, const dmfe::device_vector<double>& QRv, dmfe::device_vector<double>& out_log_slice, cudaStream_t stream) {
    const size_t offset = QRv.size() - len;
    const double* src = thrust::raw_pointer_cast(QRv.data()) + offset;
    if (out_log_slice.size() != len) out_log_slice.resize(len);
    thrust::copy_n(src, len, out_log_slice.begin());
    thrust::transform(out_log_slice.begin(), out_log_slice.end(), out_log_slice.begin(), [] __device__ (double v) { return v > 0.0 ? log(v) : v; });
}

// Host launcher using explicit precomputed log slice
void indexVecLN3GPU_log_cached(
    const dmfe::device_vector<double>& weights,
    const dmfe::device_vector<size_t>& inds,
    const dmfe::device_vector<double>& QKv,
    const dmfe::device_vector<double>& QRv,
    const dmfe::device_vector<double>& logQR_last,
    size_t len,
    dmfe::device_vector<double>& qk_result,
    dmfe::device_vector<double>& qr_result,
    cudaStream_t stream)
{
    const size_t depth = weights.size() / inds.size();
    IndexVecLN3LogOptimizer::setupKernel(depth);
    const size_t offset = QKv.size() - len;
    const double* QKv_last_ptr = thrust::raw_pointer_cast(QKv.data()) + offset;
    const double* QRv_last_ptr  = thrust::raw_pointer_cast(QRv.data()) + (QRv.size() - len);
    IndexVecLN3LogOptimizer::execute(weights, inds, QKv_last_ptr, QRv_last_ptr, logQR_last, qk_result, qr_result, stream);
}

// ================= indexVecN =====================

__global__ __launch_bounds__(64, 1)
void indexVecNKernel_optimized(
    const double* __restrict__ weights,
    const size_t* __restrict__ inds,
    const double* __restrict__ dtratio,
    const double* __restrict__ QKv,
    const double* __restrict__ QRv,
    const double* __restrict__ dQKv,
    const double* __restrict__ dQRv,
    double* __restrict__ qK_result,
    double* __restrict__ qR_result,
    size_t len,
    size_t t1len)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = len * len;
    if (tid >= total_elements) return;
    if (t1len < 2) {
        size_t j0 = tid % len;
        qK_result[tid] = QKv[j0];
        qR_result[tid] = QRv[j0];
        return;
    }
    size_t i = tid / len; size_t j = tid % len;
    double weight = weights[i]; size_t ind = inds[i];
    double weight2 = weight * weight; double weight3 = weight2 * weight;
    size_t curr_idx = ind * len + j; size_t base_idx = curr_idx - len; size_t next_idx = curr_idx + len;
    double qK_val, qR_val;
    if (ind < t1len - 1) {
        double qK_base = QKv[base_idx]; double qK_curr = QKv[curr_idx];
        double dqK_curr = dQKv[curr_idx]; double dqK_next = dQKv[next_idx];
        double dt_ratio = dtratio[ind + 1];
        double qR_base = QRv[base_idx]; double qR_curr = QRv[curr_idx];
        double dqR_curr = dQRv[curr_idx]; double dqR_next = dQRv[next_idx];
        double coeff1 = 1.0 - 3.0 * weight2 - 2.0 * weight3;
        double coeff2 = 3.0 * weight2 + 2.0 * weight3;
        double coeff3 = weight + 2.0 * weight2 + weight3;
        double coeff4 = (weight2 + weight3) / dt_ratio;
        qK_val = coeff1 * qK_base + coeff2 * qK_curr - coeff3 * dqK_curr - coeff4 * dqK_next;
        qR_val = coeff1 * qR_base + coeff2 * qR_curr - coeff3 * dqR_curr - coeff4 * dqR_next;
    } else {
        double qK_base = QKv[base_idx]; double qK_curr = QKv[curr_idx]; double dqK_curr = dQKv[curr_idx];
        double qR_base = QRv[base_idx]; double qR_curr = QRv[curr_idx]; double dqR_curr = dQRv[curr_idx];
        double coeff1 = 1.0 - weight2; double coeff2 = weight + weight2;
        qK_val = coeff1 * qK_base + weight2 * qK_curr - coeff2 * dqK_curr;
        qR_val = coeff1 * qR_base + weight2 * qR_curr - coeff2 * dqR_curr;
    }
    qK_result[tid] = qK_val; qR_result[tid] = qR_val;
}

// Log-space variant:
// QR via Hermite of fR=log(QR), gR=dQR/QR; QK interpolated in the ordinary linear domain.
__global__ __launch_bounds__(64, 1)
void indexVecNKernel_optimized_log(
    const double* __restrict__ weights,
    const size_t* __restrict__ inds,
    const double* __restrict__ dtratio,
    const double* __restrict__ QKv,
    const double* __restrict__ QRv,
    const double* __restrict__ dQKv,
    const double* __restrict__ dQRv,
    double* __restrict__ qK_result,
    double* __restrict__ qR_result,
    size_t len,
    size_t t1len)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = len * len;
    if (tid >= total_elements) return;
    if (t1len < 2) {
        size_t j0 = tid % len;
        qK_result[tid] = QKv[j0];
        qR_result[tid] = QRv[j0];
        return;
    }
    size_t i = tid / len; size_t j = tid % len;
    double weight = weights[i]; size_t ind = inds[i];
    double weight2 = weight * weight; double weight3 = weight2 * weight;
    size_t curr_idx = ind * len + j; size_t base_idx = curr_idx - len; size_t next_idx = curr_idx + len;
    // QK: ordinary linear-domain Hermite interpolation (same as non-log kernel).
    double qK_val;
    if (ind < t1len - 1) {
        double qK_base = QKv[base_idx]; double qK_curr = QKv[curr_idx];
        double dqK_curr = dQKv[curr_idx]; double dqK_next = dQKv[next_idx];
        double dt_ratio = dtratio[ind + 1];
        double coeff1 = 1.0 - 3.0 * weight2 - 2.0 * weight3;
        double coeff2 = 3.0 * weight2 + 2.0 * weight3;
        double coeff3 = weight + 2.0 * weight2 + weight3;
        double coeff4 = (weight2 + weight3) / dt_ratio;
        qK_val = coeff1 * qK_base + coeff2 * qK_curr - coeff3 * dqK_curr - coeff4 * dqK_next;
    } else {
        double qK_base = QKv[base_idx]; double qK_curr = QKv[curr_idx]; double dqK_curr = dQKv[curr_idx];
        double coeff1 = 1.0 - weight2; double coeff2 = weight + weight2;
        qK_val = coeff1 * qK_base + weight2 * qK_curr - coeff2 * dqK_curr;
    }
    // QR in log-space when values positive; else fallback to linear-domain Hermite
    double qR_val;
    if (ind < t1len - 1) {
        double QR_base = QRv[base_idx]; double QR_curr = QRv[curr_idx];
        double dqR_left = dQRv[curr_idx]; double dqR_right = dQRv[next_idx];
        bool ok = (QR_base > 0.0 && QR_curr > 0.0);
        if (ok) {
            double f_base = log(QR_base);
            double f_curr = log(QR_curr);
            // Staggered derivatives: left derivative stored at curr index pairs with QR_base; right derivative at next pairs with QR_curr
            double g_left  = dqR_left  / QR_base;
            double g_right = dqR_right / QR_curr;
            double coeff1 = 1.0 - 3.0 * weight2 - 2.0 * weight3;
            double coeff2 = 3.0 * weight2 + 2.0 * weight3;
            double coeff3 = weight + 2.0 * weight2 + weight3;
            double coeff4 = (weight2 + weight3) / dtratio[ind + 1];
            double f_interp = coeff1 * f_base + coeff2 * f_curr - coeff3 * g_left - coeff4 * g_right;
            qR_val = exp(f_interp);
        } else {
            double coeff1 = 1.0 - 3.0 * weight2 - 2.0 * weight3;
            double coeff2 = 3.0 * weight2 + 2.0 * weight3;
            double coeff3 = weight + 2.0 * weight2 + weight3;
            double coeff4 = (weight2 + weight3) / dtratio[ind + 1];
            qR_val = coeff1 * QR_base + coeff2 * QR_curr - coeff3 * dqR_left - coeff4 * dqR_right;
        }
    } else {
        double QR_base = QRv[base_idx]; double QR_curr = QRv[curr_idx]; double dqR_left = dQRv[curr_idx];
        if (QR_base > 0.0 && QR_curr > 0.0) {
            double f_base = log(QR_base);
            double f_curr = log(QR_curr);
            double g_left = dqR_left / QR_base;
            double coeff1 = 1.0 - weight2; double coeff2 = weight + weight2;
            double f_interp = coeff1 * f_base + weight2 * f_curr - coeff2 * g_left;
            qR_val = exp(f_interp);
        } else {
            double coeff1 = 1.0 - weight2; double coeff2 = weight + weight2;
            qR_val = coeff1 * QR_base + weight2 * QR_curr - coeff2 * dqR_left;
        }
    }
    qK_result[tid] = qK_val; qR_result[tid] = qR_val;
}

void indexVecNGPU(
    const dmfe::device_vector<double>& weights,
    const dmfe::device_vector<size_t>& inds,
    const dmfe::device_vector<double>& dtratio,
    dmfe::device_vector<double>& qK_result,
    dmfe::device_vector<double>& qR_result,
    const dmfe::device_vector<double>& QKv,
    const dmfe::device_vector<double>& QRv,
    const dmfe::device_vector<double>& dQKv,
    const dmfe::device_vector<double>& dQRv,
    size_t len,
    cudaStream_t stream)
{
    size_t total_elements = len * len; size_t t1len = dtratio.size();
    int threads = 64; int blocks = (total_elements + threads - 1) / threads;
    indexVecNKernel_optimized<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(weights.data()),
        thrust::raw_pointer_cast(inds.data()),
        thrust::raw_pointer_cast(dtratio.data()),
        thrust::raw_pointer_cast(QKv.data()),
        thrust::raw_pointer_cast(QRv.data()),
        thrust::raw_pointer_cast(dQKv.data()),
        thrust::raw_pointer_cast(dQRv.data()),
        thrust::raw_pointer_cast(qK_result.data()),
        thrust::raw_pointer_cast(qR_result.data()),
        len, t1len);
}

// Log-space variant launcher
void indexVecNGPU_log(
    const dmfe::device_vector<double>& weights,
    const dmfe::device_vector<size_t>& inds,
    const dmfe::device_vector<double>& dtratio,
    dmfe::device_vector<double>& qK_result,
    dmfe::device_vector<double>& qR_result,
    const dmfe::device_vector<double>& QKv,
    const dmfe::device_vector<double>& QRv,
    const dmfe::device_vector<double>& dQKv,
    const dmfe::device_vector<double>& dQRv,
    size_t len,
    cudaStream_t stream)
{
    size_t total_elements = len * len; size_t t1len = dtratio.size();
    int threads = 64; int blocks = (total_elements + threads - 1) / threads;
    indexVecNKernel_optimized_log<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(weights.data()),
        thrust::raw_pointer_cast(inds.data()),
        thrust::raw_pointer_cast(dtratio.data()),
        thrust::raw_pointer_cast(QKv.data()),
        thrust::raw_pointer_cast(QRv.data()),
        thrust::raw_pointer_cast(dQKv.data()),
        thrust::raw_pointer_cast(dQRv.data()),
        thrust::raw_pointer_cast(qK_result.data()),
        thrust::raw_pointer_cast(qR_result.data()),
        len, t1len);
}

// ================= indexVecR2 =====================

__global__ __launch_bounds__(64, 1)
void indexVecR2Kernel(
    const double* __restrict__ in1,
    const double* __restrict__ in2,
    const double* __restrict__ in3,
    const size_t* __restrict__ inds,
    const double* __restrict__ dtratio,
    double* __restrict__ result,
    size_t dims,
    size_t t1len)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= dims) return;
    double u = in3[i]; double u2 = u * u; double u3 = u2 * u; size_t ind = inds[i]; double val;
    if (ind < t1len - 1) {
        double in1m1 = in1[ind - 1]; double in10 = in1[ind]; double in20 = in2[ind]; double in21 = in2[ind + 1]; double dtr = dtratio[ind + 1];
        val = (1.0 - 3.0 * u2 - 2.0 * u3) * in1m1 + (3.0 * u2 + 2.0 * u3) * in10 - (u + 2.0 * u2 + u3) * in20 - (u2 + u3) * in21 / dtr;
    } else {
        double in1m1 = in1[ind - 1]; double in10 = in1[ind]; double in20 = in2[ind];
        val = (1.0 - u2) * in1m1 + u2 * in10 - (u + u2) * in20;
    }
    result[i] = val;
}

void indexVecR2GPU(
    const dmfe::device_vector<double>& in1,
    const dmfe::device_vector<double>& in2,
    const dmfe::device_vector<double>& in3,
    const dmfe::device_vector<size_t>& inds,
    const dmfe::device_vector<double>& dtratio,
    dmfe::device_vector<double>& result,
    cudaStream_t stream)
{
    size_t dims = inds.size(); size_t t1len = dtratio.size();
    int threads = 64; int blocks = (dims + threads - 1) / threads;
    indexVecR2Kernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(in1.data()),
        thrust::raw_pointer_cast(in2.data()),
        thrust::raw_pointer_cast(in3.data()),
        thrust::raw_pointer_cast(inds.data()),
        thrust::raw_pointer_cast(dtratio.data()),
        thrust::raw_pointer_cast(result.data()),
        dims, t1len);
}

// ================= Host Functions =====================

// GPU interpolation functions
