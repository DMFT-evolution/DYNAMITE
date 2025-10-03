#pragma once
#include "core/config_build.hpp"
#include <vector>
#include <cstddef>

#if DMFE_WITH_CUDA
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#endif

// CPU version of indexMatAll
void indexMatAll(const std::vector<double>& posx, 
                 const std::vector<size_t>& indsy,
                 const std::vector<double>& weightsy, 
                 const std::vector<double>& dtratio,
                 std::vector<double>& qK_result, 
                 std::vector<double>& qR_result);

#if DMFE_WITH_CUDA
// Forward declaration of optimizer interface
void indexMatAllGPU(const thrust::device_vector<double>& posx,
                    const thrust::device_vector<size_t>& indsy,
                    const thrust::device_vector<double>& weightsy,
                    const thrust::device_vector<double>& dtratio,
                    thrust::device_vector<double>& qK_result,
                    thrust::device_vector<double>& qR_result,
                    const thrust::device_vector<double>& QKv,
                    const thrust::device_vector<double>& QRv,
                    const thrust::device_vector<double>& dQKv,
                    const thrust::device_vector<double>& dQRv,
                    size_t len,
                    cudaStream_t stream = 0);
#endif
