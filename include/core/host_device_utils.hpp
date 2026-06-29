#pragma once
#include <vector>
#include "core/device_vector.hpp"
#include <thrust/device_ptr.h>
#include <cassert>

// Host utilities
// Element-wise product: out[i] = a[i] * b[i]
void Product(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out);
void scaleVec(const std::vector<double>& in, double s, std::vector<double>& out);
void printVectorDifference(const std::vector<double>& a, const std::vector<double>& b);

// Device utilities
dmfe::device_vector<double> SubtractGPU(const dmfe::device_vector<double>& a, const dmfe::device_vector<double>& b);
dmfe::device_vector<double> scalarMultiply(const dmfe::device_vector<double>& v, double s);
dmfe::device_vector<double> scalarMultiply_ptr(const thrust::device_ptr<double>& v, double s, size_t len);

void AddSubtractGPU(dmfe::device_vector<double>& gK,
                    const dmfe::device_vector<double>& gKfinal,
                    const dmfe::device_vector<double>& gK0,
                    dmfe::device_vector<double>& gR,
                    const dmfe::device_vector<double>& gRfinal,
                    const dmfe::device_vector<double>& gR0,
                    cudaStream_t stream = 0);

void FusedUpdate(const thrust::device_ptr<double>& a,
                 const thrust::device_ptr<double>& b,
                 const dmfe::device_vector<double>& out,
                 const double* alpha,
                 const double* beta,
                 const dmfe::device_vector<double>* delta = nullptr,
                 const dmfe::device_vector<double>* extra1 = nullptr,
                 const dmfe::device_vector<double>* extra2 = nullptr,
                 const dmfe::device_vector<double>* extra3 = nullptr,
                 const thrust::device_ptr<double>& subtract = nullptr,
                 cudaStream_t stream = 0);
