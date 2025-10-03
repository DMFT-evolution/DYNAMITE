#ifndef SPARSIFY_UTILS_HPP
#define SPARSIFY_UTILS_HPP

#include "core/config_build.hpp"
#include <vector>

#if DMFE_WITH_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#endif

// CPU sparsification and scaling function
void sparsifyNscale(double threshold);

#if DMFE_WITH_CUDA
// GPU sparsification and scaling function
void sparsifyNscaleGPU(double threshold, cudaStream_t stream = 0);

// GPU gather function
thrust::device_vector<double> gatherGPU(const thrust::device_vector<double>& v,
                                        const thrust::device_vector<size_t>& idxs,
                                        size_t len,
                                        const thrust::device_vector<double>& scale = {},
                                        cudaStream_t stream = 0);

// CUDA kernels
__global__ void gatherKernel(const double* __restrict__ v,
                            const size_t* __restrict__ idxs,
                            const double* __restrict__ scale,
                            double* __restrict__ out,
                            size_t len, size_t n_chunks, bool use_scale);

__global__ void computeSparsifyFlags(const double* __restrict__ t1grid,
                                     const double* __restrict__ QKv,
                                     const double* __restrict__ QRv,
                                     const double* __restrict__ dQKv,
                                     const double* __restrict__ dQRv,
                                     unsigned char* __restrict__ flags,
                                     double threshold, size_t len, size_t n);
#endif // DMFE_WITH_CUDA

#endif // SPARSIFY_UTILS_HPP
