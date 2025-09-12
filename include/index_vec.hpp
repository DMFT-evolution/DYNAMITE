#pragma once
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

// Host interpolation vector functions
void indexVecLN3(const std::vector<double>& weights, const std::vector<size_t>& inds,
                 std::vector<double>& qk_result, std::vector<double>& qr_result, size_t len);

void indexVecN(const size_t length, const std::vector<double>& weights, const std::vector<size_t>& inds, 
               const std::vector<double>& dtratio, std::vector<double>& qK_result, std::vector<double>& qR_result, size_t len);

void indexVecR2(const std::vector<double>& in1, const std::vector<double>& in2, const std::vector<double>& in3, 
                const std::vector<size_t>& inds, const std::vector<double>& dtratio, std::vector<double>& result);

// Host-facing GPU interpolation vector APIs. Signatures preserved for behavioral parity.

void indexVecLN3GPU(
    const thrust::device_vector<double>& weights,
    const thrust::device_vector<size_t>& inds,
    const thrust::device_vector<double>& QKv,
    const thrust::device_vector<double>& QRv,
    size_t len,
    thrust::device_vector<double>& qk_result,
    thrust::device_vector<double>& qr_result,
    cudaStream_t stream = 0);

void indexVecNGPU(
    const thrust::device_vector<double>& weights,
    const thrust::device_vector<size_t>& inds,
    const thrust::device_vector<double>& dtratio,
    thrust::device_vector<double>& qK_result,
    thrust::device_vector<double>& qR_result,
    const thrust::device_vector<double>& QKv,
    const thrust::device_vector<double>& QRv,
    const thrust::device_vector<double>& dQKv,
    const thrust::device_vector<double>& dQRv,
    size_t len,
    cudaStream_t stream);

void indexVecR2GPU(
    const thrust::device_vector<double>& in1,
    const thrust::device_vector<double>& in2,
    const thrust::device_vector<double>& in3,
    const thrust::device_vector<size_t>& inds,
    const thrust::device_vector<double>& dtratio,
    thrust::device_vector<double>& result,
    cudaStream_t stream);
