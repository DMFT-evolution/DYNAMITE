#include "convolution.hpp"
#include "globals.hpp"
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

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

thrust::device_vector<double> ConvAGPU(const thrust::device_vector<double>& f,
                                       const thrust::device_vector<double>& g,
                                       double t,
                                       const thrust::device_vector<double>& integ,
                                       const thrust::device_vector<double>& theta,
                                       cudaStream_t stream) {
    size_t length = integ.size(); size_t depth = f.size() / length;
    thrust::device_vector<double> out(depth, 0.0);
    int threads = 64; size_t shmem = threads * sizeof(double);
    ConvAGPUKernel<<<depth, threads, shmem, stream>>>(
        thrust::raw_pointer_cast(f.data()),
        thrust::raw_pointer_cast(g.data()),
        thrust::raw_pointer_cast(integ.data()),
        (theta.size()==depth? thrust::raw_pointer_cast(theta.data()): nullptr),
        thrust::raw_pointer_cast(out.data()), t, length, depth);
    return out;
}

thrust::device_vector<double> ConvAGPU(const thrust::device_vector<double>& f,
                                       const thrust::device_ptr<double>& g,
                                       double t,
                                       const thrust::device_vector<double>& integ,
                                       const thrust::device_vector<double>& theta,
                                       cudaStream_t stream) {
    size_t length = integ.size(); size_t depth = f.size() / length;
    thrust::device_vector<double> out(depth, 0.0);
    int threads = 64; size_t shmem = threads * sizeof(double);
    ConvAGPUKernel<<<depth, threads, shmem, stream>>>(
        thrust::raw_pointer_cast(f.data()),
        thrust::raw_pointer_cast(g),
        thrust::raw_pointer_cast(integ.data()),
        (theta.size()==depth? thrust::raw_pointer_cast(theta.data()): nullptr),
        thrust::raw_pointer_cast(out.data()), t, length, depth);
    return out;
}

void ConvAGPU_Stream(const thrust::device_vector<double>& f,
                     const thrust::device_vector<double>& g,
                     thrust::device_vector<double>& out,
                     thrust::device_vector<double>& t,
                     const thrust::device_vector<double>& integ,
                     const thrust::device_vector<double>& theta,
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

void ConvAGPU_Stream(const thrust::device_vector<double>& f,
                     const thrust::device_ptr<double>& g,
                     thrust::device_vector<double>& out,
                     double t,
                     const thrust::device_vector<double>& integ,
                     const thrust::device_vector<double>& theta,
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

thrust::device_vector<double> ConvRGPU(const thrust::device_vector<double>& f,
                                       const thrust::device_vector<double>& g,
                                       double t,
                                       const thrust::device_vector<double>& integ,
                                       const thrust::device_vector<double>& theta,
                                       cudaStream_t stream) {
    size_t length = integ.size(); size_t depth = f.size() / length;
    thrust::device_vector<double> out(depth, 0.0);
    int threads = 64; size_t shmem = length * sizeof(double) + threads * sizeof(double);
    ConvRKernel<<<depth, threads, shmem, stream>>>(
        thrust::raw_pointer_cast(f.data()),
        thrust::raw_pointer_cast(g.data()),
        thrust::raw_pointer_cast(integ.data()),
        thrust::raw_pointer_cast(theta.data()),
        thrust::raw_pointer_cast(out.data()), t, length, depth);
    return out;
}

void ConvRGPU_Stream(const thrust::device_vector<double>& f,
                     const thrust::device_vector<double>& g,
                     thrust::device_vector<double>& out,
                     const thrust::device_vector<double>& t,
                     const thrust::device_vector<double>& integ,
                     const thrust::device_vector<double>& theta,
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
std::vector<double> ConvA(const std::vector<double>& f, const std::vector<double>& g, const double t)
{
    size_t length = sim->h_integ.size();
    size_t depth = f.size() / length;
    std::vector<double> out(depth, 0.0);
    if (depth == 1)
    {
        double temp = 0.0;
        for (size_t j = 0; j < length; j++)
        {
            temp += t * sim->h_integ[j] * f[j] * g[j];
        }
        out[0] = temp;
    }
    else
    {
        for (size_t j = 0; j < depth; j++)
        {
            for (size_t i = 0; i < length; i++)
            {
                out[j] += sim->h_integ[i] * f[j * length + i] * g[j * length + i];
            }
            out[j] *= t * sim->h_theta[j];
        }
    }
    return out;
}

std::vector<double> ConvR(const std::vector<double>& f, const std::vector<double>& g, const double t)
{
    size_t length = sim->h_integ.size();
    size_t depth = f.size() / length;
    std::vector<double> out(length, 0.0);
    for (size_t j = 0; j < length; j++)
    {
        for (size_t i = 0; i < depth; i++)
        {
            out[j] += sim->h_integ[i] * f[j * length + i] * g[j * length + i];
        }
        out[j] *= t * (1 - sim->h_theta[j]);
    }
    return out;
}
