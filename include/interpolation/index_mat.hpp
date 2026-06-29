#pragma once
#include "core/config_build.hpp"
#include <vector>
#include <cstddef>

#if DMFE_WITH_CUDA
#include "core/device_vector.cuh"
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
void indexMatAllGPU(const dmfe::device_vector<double>& posx,
                    const dmfe::device_vector<size_t>& indsy,
                    const dmfe::device_vector<double>& weightsy,
                    const dmfe::device_vector<double>& dtratio,
                    dmfe::device_vector<double>& qK_result,
                    dmfe::device_vector<double>& qR_result,
                    const dmfe::device_vector<double>& QKv,
                    const dmfe::device_vector<double>& QRv,
                    const dmfe::device_vector<double>& dQKv,
                    const dmfe::device_vector<double>& dQRv,
                    size_t len,
                    cudaStream_t stream = 0);

void indexMatAllGPU_log(const dmfe::device_vector<double>& posx,
                    const dmfe::device_vector<size_t>& indsy,
                    const dmfe::device_vector<double>& weightsy,
                    const dmfe::device_vector<double>& dtratio,
                    dmfe::device_vector<double>& qK_result,
                    dmfe::device_vector<double>& qR_result,
                    const dmfe::device_vector<double>& QKv,
                    const dmfe::device_vector<double>& QRv,
                    const dmfe::device_vector<double>& dQKv,
                    const dmfe::device_vector<double>& dQRv,
                    size_t len,
                    cudaStream_t stream = 0);
#endif
