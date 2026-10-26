#include "convolution/convolution.cuh"
#include "core/globals.hpp"

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

// ============================================================================
// [PERF] Batched ConvA/ConvR kernel -- see the note in convolution.cuh.
// grid = (depth, n): blockIdx.x selects the output element exactly as before,
// blockIdx.y selects which of the n convolutions this block belongs to.
// The body of each branch is a verbatim copy of the corresponding single
// convolution kernel, so every output value is produced by an identical sequence
// of floating-point operations.
// ============================================================================
__global__ void ConvBatchedKernel(ConvBatchDesc batch,
                                  const double* __restrict__ integ,
                                  const double* __restrict__ theta,
                                  const double* __restrict__ t,
                                  size_t length)
{
    extern __shared__ double shared[];
    const int j   = blockIdx.x;
    const int k   = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const double* __restrict__ f   = batch.f[k];
    const double* __restrict__ g   = batch.g[k];
    double* __restrict__       out = batch.out[k];
    const size_t base = static_cast<size_t>(j) * length;

    if (batch.kind[k] == 0) {
        // --- ConvA: identical to ConvAGPUKernel -------------------------------
        double* sdata = shared;
        double sum = 0.0;
        for (size_t i = tid; i < length; i += nthreads) {
            sum += integ[i] * f[base + i] * g[base + i];
        }
        sdata[tid] = sum;
        __syncthreads();
        for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) sdata[tid] += sdata[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            double scale = t[0] * theta[j];
            out[j] = scale * sdata[0];
        }
    } else {
        // --- ConvR: identical to ConvRKernel ----------------------------------
        double* integ_shared     = shared;
        double* reduction_shared = &shared[length];
        for (int i = tid; i < length; i += nthreads) integ_shared[i] = integ[i];
        __syncthreads();
        double sum = 0.0;
        for (size_t i = tid; i < length; i += nthreads) sum += f[base + i] * g[base + i] * integ_shared[i];
        reduction_shared[tid] = sum;
        __syncthreads();
        for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduction_shared[tid] += reduction_shared[tid + stride];
            __syncthreads();
        }
        if (tid == 0) out[j] = t[0] * (1.0 - theta[j]) * reduction_shared[0];
    }
}

void ConvBatchedGPU(const ConvBatchDesc& batch,
                    const dmfe::device_vector<double>& integ,
                    const dmfe::device_vector<double>& theta,
                    const dmfe::device_vector<double>& t,
                    size_t length,
                    size_t depth,
                    cudaStream_t stream)
{
    if (batch.n == 0) return;
    const int threads = 64;                       // must match the single-kernel value
    const size_t shmem = (length + threads) * sizeof(double);   // max of the two paths
    dim3 grid(static_cast<unsigned>(depth), static_cast<unsigned>(batch.n));
    ConvBatchedKernel<<<grid, threads, shmem, stream>>>(
        batch,
        thrust::raw_pointer_cast(integ.data()),
        thrust::raw_pointer_cast(theta.data()),
        thrust::raw_pointer_cast(t.data()),
        length);
}

// [PERF] see ConvAScalarBatchDesc in convolution/convolution.cuh
__global__ void ConvABatchedScalarKernel(ConvAScalarBatchDesc batch,
                                         const double* __restrict__ integ,
                                         const double* __restrict__ theta,
                                         size_t length,
                                         size_t depth)
{
    extern __shared__ double sdata[];
    const int k = blockIdx.y;
    int j = blockIdx.x;
    int tid = threadIdx.x;

    const double* __restrict__ f   = batch.f[k];
    const double* __restrict__ g   = batch.g[k];
    double* __restrict__       out = batch.out[k];

    size_t start = j * length;
    double sum = 0.0;
    for (size_t i = tid; i < length; i += blockDim.x) {
        sum += integ[i] * f[start + i] * g[start + i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        double scale = batch.t[k] * ((depth == 1) ? 1.0 : theta[j]);
        out[j] = scale * sdata[0];
    }
}

void ConvABatchedScalarGPU(const ConvAScalarBatchDesc& batch,
                           const dmfe::device_vector<double>& integ,
                           const dmfe::device_vector<double>& theta,
                           size_t length,
                           size_t depth,
                           cudaStream_t stream)
{
    if (batch.n == 0) return;
    const int threads = 64;                      // must match ConvAGPUKernel
    const size_t shmem = threads * sizeof(double);
    dim3 grid(static_cast<unsigned>(depth), static_cast<unsigned>(batch.n));
    ConvABatchedScalarKernel<<<grid, threads, shmem, stream>>>(
        batch,
        thrust::raw_pointer_cast(integ.data()),
        (theta.size() == depth ? thrust::raw_pointer_cast(theta.data()) : nullptr),
        length, depth);
}

// ---- ConvA kernels ----
__global__ void ConvAGPUKernel(const double* __restrict__ f,
                               const double* __restrict__ g,
                               const double* __restrict__ integ,
                               const double* __restrict__ theta,
                               double* __restrict__ out,
                               double t,
                               size_t length,
                               size_t depth) {
    extern __shared__ double sdata[];
    int j = blockIdx.x;
    int tid = threadIdx.x;
    size_t start = j * length;
    double sum = 0.0;
    for (size_t i = tid; i < length; i += blockDim.x) {
        sum += integ[i] * f[start + i] * g[start + i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        double scale = t * ((depth == 1) ? 1.0 : theta[j]);
        out[j] = scale * sdata[0];
    }
}

__global__ void ConvAGPUKernel(const double* __restrict__ f,
                               const double* __restrict__ g,
                               const double* __restrict__ integ,
                               const double* __restrict__ theta,
                               double* __restrict__ out,
                               const double* __restrict__ t,
                               size_t length,
                               size_t depth) {
    extern __shared__ double sdata[];
    int j = blockIdx.x;
    int tid = threadIdx.x;
    size_t start = j * length;
    double sum = 0.0;
    for (size_t i = tid; i < length; i += blockDim.x) {
        sum += integ[i] * f[start + i] * g[start + i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        double scale = t[0] * ((depth == 1) ? 1.0 : theta[j]);
        out[j] = scale * sdata[0];
    }
}

dmfe::device_vector<double> ConvAGPU(const dmfe::device_vector<double>& f,
                                       const dmfe::device_vector<double>& g,
                                       double t,
                                       const dmfe::device_vector<double>& integ,
                                       const dmfe::device_vector<double>& theta,
                                       cudaStream_t stream) {
    size_t length = integ.size(); size_t depth = f.size() / length;
    dmfe::device_vector<double> out(depth, 0.0, dmfe::cuda_async_allocator<double>(stream));
    int threads = 64; size_t shmem = threads * sizeof(double);
    ConvAGPUKernel<<<depth, threads, shmem, stream>>>(
        thrust::raw_pointer_cast(f.data()),
        thrust::raw_pointer_cast(g.data()),
        thrust::raw_pointer_cast(integ.data()),
        (theta.size()==depth? thrust::raw_pointer_cast(theta.data()): nullptr),
        thrust::raw_pointer_cast(out.data()), t, length, depth);
    return out;
}

dmfe::device_vector<double> ConvAGPU(const dmfe::device_vector<double>& f,
                                       const thrust::device_ptr<double>& g,
                                       double t,
                                       const dmfe::device_vector<double>& integ,
                                       const dmfe::device_vector<double>& theta,
                                       cudaStream_t stream) {
    size_t length = integ.size(); size_t depth = f.size() / length;
    dmfe::device_vector<double> out(depth, 0.0, dmfe::cuda_async_allocator<double>(stream));
    int threads = 64; size_t shmem = threads * sizeof(double);
    ConvAGPUKernel<<<depth, threads, shmem, stream>>>(
        thrust::raw_pointer_cast(f.data()),
        thrust::raw_pointer_cast(g),
        thrust::raw_pointer_cast(integ.data()),
        (theta.size()==depth? thrust::raw_pointer_cast(theta.data()): nullptr),
        thrust::raw_pointer_cast(out.data()), t, length, depth);
    return out;
}

void ConvAGPU_Stream(const dmfe::device_vector<double>& f,
                     const dmfe::device_vector<double>& g,
                     dmfe::device_vector<double>& out,
                     dmfe::device_vector<double>& t,
                     const dmfe::device_vector<double>& integ,
                     const dmfe::device_vector<double>& theta,
                     cudaStream_t stream) {
    size_t length = integ.size(); size_t depth = f.size() / length;
    int threads = 64; size_t shmem = threads * sizeof(double);
    ConvAGPUKernel<<<depth, threads, shmem, stream>>>(
        thrust::raw_pointer_cast(f.data()),
        thrust::raw_pointer_cast(g.data()),
        thrust::raw_pointer_cast(integ.data()),
        thrust::raw_pointer_cast(theta.data()),
        thrust::raw_pointer_cast(out.data()),
        thrust::raw_pointer_cast(t.data()), length, depth);
}

void ConvAGPU_Stream(const dmfe::device_vector<double>& f,
                     const thrust::device_ptr<double>& g,
                     dmfe::device_vector<double>& out,
                     double t,
                     const dmfe::device_vector<double>& integ,
                     const dmfe::device_vector<double>& theta,
                     cudaStream_t stream) {
    size_t length = integ.size(); size_t depth = f.size() / length;
    int threads = 64; size_t shmem = threads * sizeof(double);
    ConvAGPUKernel<<<depth, threads, shmem, stream>>>(
        thrust::raw_pointer_cast(f.data()),
        thrust::raw_pointer_cast(g),
        thrust::raw_pointer_cast(integ.data()),
        (theta.size()==depth? thrust::raw_pointer_cast(theta.data()): nullptr),
        thrust::raw_pointer_cast(out.data()), t, length, depth);
}

// Const-pointer overload
void ConvAGPU_Stream(const dmfe::device_vector<double>& f,
                     const thrust::device_ptr<const double>& g,
                     dmfe::device_vector<double>& out,
                     double t,
                     const dmfe::device_vector<double>& integ,
                     const dmfe::device_vector<double>& theta,
                     cudaStream_t stream) {
    size_t length = integ.size(); size_t depth = f.size() / length;
    int threads = 64; size_t shmem = threads * sizeof(double);
    ConvAGPUKernel<<<depth, threads, shmem, stream>>>(
        thrust::raw_pointer_cast(f.data()),
        thrust::raw_pointer_cast(g),
        thrust::raw_pointer_cast(integ.data()),
        (theta.size()==depth? thrust::raw_pointer_cast(theta.data()): nullptr),
        thrust::raw_pointer_cast(out.data()), t, length, depth);
}

// ---- ConvR kernels ----
__global__ void ConvRKernel(const double* __restrict__ f,
                            const double* __restrict__ g,
                            const double* __restrict__ integ,
                            const double* __restrict__ theta,
                            double* __restrict__ out,
                            double t,
                            size_t length,
                            size_t depth) {
    extern __shared__ double shared[];
    double* integ_shared = shared;
    double* reduction_shared = &shared[length];
    int j = blockIdx.x; int tid = threadIdx.x; int nthreads = blockDim.x;
    for (int i = tid; i < length; i += nthreads) integ_shared[i] = integ[i];
    __syncthreads();
    double sum = 0.0; size_t base = j * length;
    for (size_t i = tid; i < length; i += nthreads) sum += f[base + i] * g[base + i] * integ_shared[i];
    reduction_shared[tid] = sum; __syncthreads();
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) { if (tid < stride) reduction_shared[tid] += reduction_shared[tid + stride]; __syncthreads(); }
    if (tid == 0) out[j] = t * (1.0 - theta[j]) * reduction_shared[0];
}

__global__ void ConvRKernel(const double* __restrict__ f,
                            const double* __restrict__ g,
                            const double* __restrict__ integ,
                            const double* __restrict__ theta,
                            double* __restrict__ out,
                            const double* __restrict__ t,
                            size_t length,
                            size_t depth) {
    extern __shared__ double shared[];
    double* integ_shared = shared;
    double* reduction_shared = &shared[length];
    int j = blockIdx.x; int tid = threadIdx.x; int nthreads = blockDim.x;
    for (int i = tid; i < length; i += nthreads) integ_shared[i] = integ[i];
    __syncthreads();
    double sum = 0.0; size_t base = j * length;
    for (size_t i = tid; i < length; i += nthreads) sum += f[base + i] * g[base + i] * integ_shared[i];
    reduction_shared[tid] = sum; __syncthreads();
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) { if (tid < stride) reduction_shared[tid] += reduction_shared[tid + stride]; __syncthreads(); }
    if (tid == 0) out[j] = t[0] * (1.0 - theta[j]) * reduction_shared[0];
}

dmfe::device_vector<double> ConvRGPU(const dmfe::device_vector<double>& f,
                                       const dmfe::device_vector<double>& g,
                                       double t,
                                       const dmfe::device_vector<double>& integ,
                                       const dmfe::device_vector<double>& theta,
                                       cudaStream_t stream) {
    size_t length = integ.size(); size_t depth = f.size() / length;
    dmfe::device_vector<double> out(depth, 0.0, dmfe::cuda_async_allocator<double>(stream));
    int threads = 64; size_t shmem = length * sizeof(double) + threads * sizeof(double);
    ConvRKernel<<<depth, threads, shmem, stream>>>(
        thrust::raw_pointer_cast(f.data()),
        thrust::raw_pointer_cast(g.data()),
        thrust::raw_pointer_cast(integ.data()),
        thrust::raw_pointer_cast(theta.data()),
        thrust::raw_pointer_cast(out.data()), t, length, depth);
    return out;
}

void ConvRGPU_Stream(const dmfe::device_vector<double>& f,
                     const dmfe::device_vector<double>& g,
                     dmfe::device_vector<double>& out,
                     const dmfe::device_vector<double>& t,
                     const dmfe::device_vector<double>& integ,
                     const dmfe::device_vector<double>& theta,
                     cudaStream_t stream) {
    size_t length = integ.size(); size_t depth = f.size() / length;
    int threads = 64; size_t shmem = length * sizeof(double) + threads * sizeof(double);
    ConvRKernel<<<depth, threads, shmem, stream>>>(
        thrust::raw_pointer_cast(f.data()),
        thrust::raw_pointer_cast(g.data()),
        thrust::raw_pointer_cast(integ.data()),
        thrust::raw_pointer_cast(theta.data()),
        thrust::raw_pointer_cast(out.data()),
        thrust::raw_pointer_cast(t.data()), length, depth);
}

// Host convolution functions
// GPU convolution functions
