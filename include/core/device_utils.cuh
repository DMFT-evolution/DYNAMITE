#pragma once
#include "core/config_build.hpp"
#include <cmath>
#include <string>
#include <cstdio>

#if DMFE_WITH_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

// Debug-only CUDA error check utilities. Use tiny inline helpers + macros to avoid overhead when not debugging.
inline void __dmfe_cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", expr, file, line, cudaGetErrorString(err));
        abort();
    }
}

// Use only when config.debug is true in call sites; otherwise skip to avoid sync costs.
#define DMFE_CUDA_CHECK(expr) __dmfe_cuda_check((expr), #expr, __FILE__, __LINE__)

// Post-kernel launch check: call cudaGetLastError() and optionally sync in debug paths.
inline void __dmfe_cuda_post_launch_check(const char* where, const char* file, int line) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failure at %s (%s:%d): %s\n", where, file, line, cudaGetErrorString(e));
        abort();
    }
}
#define DMFE_CUDA_POSTLAUNCH(where) __dmfe_cuda_post_launch_check(where, __FILE__, __LINE__)
#endif

// Fallback no-op definitions to keep call sites compilable when CUDA is off
#ifndef DMFE_CUDA_CHECK
#define DMFE_CUDA_CHECK(expr) ((void)0)
#endif
#ifndef DMFE_CUDA_POSTLAUNCH
#define DMFE_CUDA_POSTLAUNCH(where) ((void)0)
#endif

// Device capability check
bool isCompatibleGPUInstalled();

// System utility functions
bool isHDF5Available();
size_t getCurrentMemoryUsage();
size_t getGPUMemoryUsage();
size_t getAvailableGPUMemory();
void updatePeakMemory();
std::string getHostname();

// External global variables for peak memory tracking
extern size_t peak_memory_kb;
extern size_t peak_gpu_memory_mb;

#if DMFE_WITH_CUDA
// Device vector helpers (implemented in device_utils.cu)
thrust::device_vector<double> SubtractGPU(const thrust::device_vector<double>& a, const thrust::device_vector<double>& b);
thrust::device_vector<double> scalarMultiply(const thrust::device_vector<double>& vec, double scalar);
thrust::device_vector<double> scalarMultiply_ptr(const thrust::device_ptr<double>& vec, double scalar, size_t len);
void AddSubtractGPU(thrust::device_vector<double>& gK,
                    const thrust::device_vector<double>& gKfinal,
                    const thrust::device_vector<double>& gK0,
                    thrust::device_vector<double>& gR,
                    const thrust::device_vector<double>& gRfinal,
                    const thrust::device_vector<double>& gR0,
                    cudaStream_t stream = 0);

void FusedUpdate(const thrust::device_ptr<double>& a,
                 const thrust::device_ptr<double>& b,
                 const thrust::device_vector<double>& out,
                 const double* alpha,
                 const double* beta,
                 const thrust::device_vector<double>* delta = nullptr,
                 const thrust::device_vector<double>* extra1 = nullptr,
                 const thrust::device_vector<double>* extra2 = nullptr,
                 const thrust::device_vector<double>* extra3 = nullptr,
                 const thrust::device_ptr<double>& subtract = nullptr,
                 cudaStream_t stream = 0);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ inline double atomicAdd_double(double* address, double val) {
    unsigned long long* addr = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *addr, assumed;
    do {
        assumed = old;
        double sum = __longlong_as_double(assumed) + val;
        old = atomicCAS(addr, assumed, __double_as_longlong(sum));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#define ATOMIC_ADD_DBL(addr, v) atomicAdd_double(addr, v)
#else
#define ATOMIC_ADD_DBL(addr, v) atomicAdd(addr, v)
#endif

#endif // DMFE_WITH_CUDA

// (min/max helpers provided by math_ops.hpp)
