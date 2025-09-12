#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <string>

// Device capability check
bool isCompatibleGPUInstalled();

// System utility functions
bool isHDF5Available();
size_t getCurrentMemoryUsage();
size_t getGPUMemoryUsage();
void updatePeakMemory();
std::string getHostname();

// External global variables for peak memory tracking
extern size_t peak_memory_kb;
extern size_t peak_gpu_memory_mb;

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

// (min/max helpers provided by math_ops.hpp)
