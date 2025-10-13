#include "core/device_utils.cuh"
#include "core/config.hpp"
#include <sys/resource.h>
#include <unistd.h>
#include <chrono>
#include "core/console.hpp"
#include <cstdlib>
#include <sys/sysinfo.h>
#if defined(H5_RUNTIME_OPTIONAL)
#include "io/h5_runtime.hpp"
#endif

__global__ static void update_gK_gR_kernel(
	double* gK, const double* gKfinal, const double* gK0,
	double* gR, const double* gRfinal, const double* gR0, int N) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < N) {
		gK[i] += gK0[i] - gKfinal[i];
		gR[i] += gR0[i] - gRfinal[i];
	}
}

thrust::device_vector<double> SubtractGPU(const thrust::device_vector<double>& a, const thrust::device_vector<double>& b) {
	thrust::device_vector<double> result(a.size());
	thrust::transform(a.begin(), a.end(), b.begin(), result.begin(), thrust::minus<double>());
	return result;
}

thrust::device_vector<double> scalarMultiply(const thrust::device_vector<double>& vec, double scalar) {
	thrust::device_vector<double> result(vec.size());
	thrust::transform(vec.begin(), vec.end(), result.begin(), [scalar] __device__(double x) { return x * scalar; });
	return result;
}

thrust::device_vector<double> scalarMultiply_ptr(const thrust::device_ptr<double>& vec, double scalar, size_t len) {
	thrust::device_vector<double> result(len);
	thrust::transform(vec, vec + len, result.begin(), [scalar] __device__(double x) { return x * scalar; });
	return result;
}

void AddSubtractGPU(thrust::device_vector<double>& gK,
					const thrust::device_vector<double>& gKfinal,
					const thrust::device_vector<double>& gK0,
					thrust::device_vector<double>& gR,
					const thrust::device_vector<double>& gRfinal,
					const thrust::device_vector<double>& gR0,
					cudaStream_t stream) {
	int N = static_cast<int>(gK.size());
	int blockSize = 128;
	int numBlocks = (N + blockSize - 1) / blockSize;
	update_gK_gR_kernel<<<numBlocks, blockSize, 0, stream>>>(
		thrust::raw_pointer_cast(gK.data()),
		thrust::raw_pointer_cast(gKfinal.data()),
		thrust::raw_pointer_cast(gK0.data()),
		thrust::raw_pointer_cast(gR.data()),
		thrust::raw_pointer_cast(gRfinal.data()),
		thrust::raw_pointer_cast(gR0.data()),
		N);
}

__global__ static void FusedUpdateKernel(
	const double* __restrict__ a,
	const double* __restrict__ b,
	const double* __restrict__ extra1,
	const double* __restrict__ extra2,
	const double* __restrict__ extra3,
	const double* __restrict__ delta,
	const double* __restrict__ subtract,
	double* __restrict__ out,
	const double* alpha,
	const double* beta,
	size_t N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	double result = alpha[0] * a[i] + beta[0] * b[i];
	if (extra1) result += extra1[i];
	if (extra2) result += extra2[i];
	if (extra3) result += extra3[i];
	if (subtract) result -= delta[i] * subtract[i];
	out[i] = result;
}

void FusedUpdate(const thrust::device_ptr<double>& a,
				 const thrust::device_ptr<double>& b,
				 const thrust::device_vector<double>& out,
				 const double* alpha,
				 const double* beta,
				 const thrust::device_vector<double>* delta,
				 const thrust::device_vector<double>* extra1,
				 const thrust::device_vector<double>* extra2,
				 const thrust::device_vector<double>* extra3,
				 const thrust::device_ptr<double>& subtract,
				 cudaStream_t stream) {
	size_t N = out.size();
	int threads = 64;
	int blocks = (N + threads - 1) / threads;
	FusedUpdateKernel<<<blocks, threads, 0, stream>>>(
		thrust::raw_pointer_cast(a),
		thrust::raw_pointer_cast(b),
		extra1 ? thrust::raw_pointer_cast(extra1->data()) : nullptr,
		extra2 ? thrust::raw_pointer_cast(extra2->data()) : nullptr,
		extra3 ? thrust::raw_pointer_cast(extra3->data()) : nullptr,
		delta ? thrust::raw_pointer_cast(delta->data()) : nullptr,
		subtract ? thrust::raw_pointer_cast(subtract) : nullptr,
		thrust::raw_pointer_cast(const_cast<thrust::device_vector<double>&>(out).data()),
		alpha, beta, N);
}

bool isCompatibleGPUInstalled() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
	std::cerr << dmfe::console::ERR() << "No CUDA-capable devices found." << std::endl;
        return false;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        // Check compute capability
        if (deviceProp.major > 8 || (deviceProp.major == 8 && deviceProp.minor >= 6)) {
			std::cout << dmfe::console::INFO() << "Compatible GPU found: " << deviceProp.name << std::endl;
            return true;
        }
    }

	std::cerr << dmfe::console::WARN() << "No compatible GPU found (compute capability >= 8.6)." << std::endl;
    return false;
}

// External declarations
extern SimulationConfig config;
extern size_t peak_memory_kb;
extern size_t peak_gpu_memory_mb;

// Function to get GPU memory usage in MB
size_t getGPUMemoryUsage() {
    if (!config.gpu) return 0;
    
    size_t free_mem, total_mem, used_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        return 0;
    }
    used_mem = total_mem - free_mem;
    return used_mem / (1024 * 1024); // Convert to MB
}

// Function to get available GPU memory in MB
size_t getAvailableGPUMemory() {
    if (!config.gpu) return 0;
    
    // Check for SLURM memory limit
    const char* slurm_mem = getenv("SLURM_MEM_PER_GPU");
    if (slurm_mem) {
        return std::atoi(slurm_mem); // Assume MB
    }
    
    // Otherwise, use CUDA total memory
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        return 0;
    }
    return total_mem / (1024 * 1024); // Convert to MB
}

// getTotalSystemMemoryKB is implemented in device_utils.cpp; ensure symbol visible here via header.

// Function to update peak memory usage
void updatePeakMemory() {
    size_t current_mem = getCurrentMemoryUsage();
    if (current_mem > peak_memory_kb) {
        peak_memory_kb = current_mem;
    }
    
    if (config.gpu) {
        size_t current_gpu_mem = getGPUMemoryUsage();
        if (current_gpu_mem > peak_gpu_memory_mb) {
            peak_gpu_memory_mb = current_gpu_mem;
        }
    }
}

