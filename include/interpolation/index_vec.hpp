#pragma once
#include "core/config_build.hpp"
#include <cstddef>
#include <vector>

#if DMFE_WITH_CUDA
#include "core/device_vector.cuh"
#include <thrust/functional.h>
#include <cuda_runtime.h>
#endif

// Host interpolation vector functions
void indexVecLN3(const std::vector<double>& weights, const std::vector<size_t>& inds,
                 std::vector<double>& qk_result, std::vector<double>& qr_result, size_t len);

void indexVecN(const size_t length, const std::vector<double>& weights, const std::vector<size_t>& inds, 
               const std::vector<double>& dtratio, std::vector<double>& qK_result, std::vector<double>& qR_result, size_t len);

void indexVecR2(const std::vector<double>& in1, const std::vector<double>& in2, const std::vector<double>& in3, 
                const std::vector<size_t>& inds, const std::vector<double>& dtratio, std::vector<double>& result);

#if DMFE_WITH_CUDA
// Host-facing GPU interpolation vector APIs. Signatures preserved for behavioral parity.

void indexVecLN3GPU(
    const dmfe::device_vector<double>& weights,
    const dmfe::device_vector<size_t>& inds,
    const dmfe::device_vector<double>& QKv,
    const dmfe::device_vector<double>& QRv,
    size_t len,
    dmfe::device_vector<double>& qk_result,
    dmfe::device_vector<double>& qr_result,
    cudaStream_t stream = 0);

// Precompute logs of the last `len` entries of QRv into provided device buffer (positive -> log, non-positive -> original)
void prepareLN3LogSliceGPU_into(size_t len,
    const dmfe::device_vector<double>& QRv,
    dmfe::device_vector<double>& out_log_slice,
    cudaStream_t stream = 0);

// Log-space LN3 using an explicitly provided precomputed log slice buffer
void indexVecLN3GPU_log_cached(
    const dmfe::device_vector<double>& weights,
    const dmfe::device_vector<size_t>& inds,
    const dmfe::device_vector<double>& QKv,
    const dmfe::device_vector<double>& QRv,
    const dmfe::device_vector<double>& logQR_last,
    size_t len,
    dmfe::device_vector<double>& qk_result,
    dmfe::device_vector<double>& qr_result,
    cudaStream_t stream = 0);

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
    cudaStream_t stream);

// Log-space variant for QR
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
    cudaStream_t stream);

void indexVecR2GPU(
    const dmfe::device_vector<double>& in1,
    const dmfe::device_vector<double>& in2,
    const dmfe::device_vector<double>& in3,
    const dmfe::device_vector<size_t>& inds,
    const dmfe::device_vector<double>& dtratio,
    dmfe::device_vector<double>& result,
    cudaStream_t stream);
#endif
