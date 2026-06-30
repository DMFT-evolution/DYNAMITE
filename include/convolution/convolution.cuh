#pragma once
#include "convolution/convolution.hpp"

#if DMFE_WITH_CUDA
#include "core/device_vector.hpp"
#include <thrust/device_ptr.h>
#include <cuda_runtime.h>

// ConvA (QK-style) GPU interfaces
dmfe::device_vector<double> ConvAGPU(const dmfe::device_vector<double>& f,
                                       const dmfe::device_vector<double>& g,
                                       double t,
                                       const dmfe::device_vector<double>& integ,
                                       const dmfe::device_vector<double>& theta,
                                       cudaStream_t stream = 0);

dmfe::device_vector<double> ConvAGPU(const dmfe::device_vector<double>& f,
                                       const thrust::device_ptr<double>& g,
                                       double t,
                                       const dmfe::device_vector<double>& integ,
                                       const dmfe::device_vector<double>& theta,
                                       cudaStream_t stream = 0);

void ConvAGPU_Stream(const dmfe::device_vector<double>& f,
                     const dmfe::device_vector<double>& g,
                     dmfe::device_vector<double>& out,
                     dmfe::device_vector<double>& t,
                     const dmfe::device_vector<double>& integ,
                     const dmfe::device_vector<double>& theta,
                     cudaStream_t stream = 0);

void ConvAGPU_Stream(const dmfe::device_vector<double>& f,
                     const thrust::device_ptr<double>& g,
                     dmfe::device_vector<double>& out,
                     double t,
                     const dmfe::device_vector<double>& integ,
                     const dmfe::device_vector<double>& theta,
                     cudaStream_t stream = 0);

// Const-pointer overload for ConvA stream variant
void ConvAGPU_Stream(const dmfe::device_vector<double>& f,
                     const thrust::device_ptr<const double>& g,
                     dmfe::device_vector<double>& out,
                     double t,
                     const dmfe::device_vector<double>& integ,
                     const dmfe::device_vector<double>& theta,
                     cudaStream_t stream = 0);

// ConvR (QR-style) GPU interfaces
dmfe::device_vector<double> ConvRGPU(const dmfe::device_vector<double>& f,
                                       const dmfe::device_vector<double>& g,
                                       double t,
                                       const dmfe::device_vector<double>& integ,
                                       const dmfe::device_vector<double>& theta,
                                       cudaStream_t stream = 0);

void ConvRGPU_Stream(const dmfe::device_vector<double>& f,
                     const dmfe::device_vector<double>& g,
                     dmfe::device_vector<double>& out,
                     const dmfe::device_vector<double>& t,
                     const dmfe::device_vector<double>& integ,
                     const dmfe::device_vector<double>& theta,
                     cudaStream_t stream = 0);
#endif
