#pragma once

#include "core/config_build.hpp"
#include <cstddef>

#if DMFE_WITH_CUDA
#include <thrust/device_ptr.h>
class StreamPool;

// Previous in-place weak smoothing filter (GPU). Uses config.filter_strength (0 disables).
void filter_old(thrust::device_ptr<double> gK,
                thrust::device_ptr<double> gR,
                thrust::device_ptr<double> hK0,
                thrust::device_ptr<double> hR0,
                size_t len,
                StreamPool& pool);

// In-place weak smoothing filter (GPU). Uses config.filter_strength (0 disables).
void filter(thrust::device_ptr<double> gK,
            thrust::device_ptr<double> gR,
            thrust::device_ptr<double> hK0,
            thrust::device_ptr<double> hR0,
            size_t len,
            StreamPool& pool);
void filter_dR(thrust::device_ptr<double> hR0,
                size_t len,
                StreamPool& pool);
void filter_dRK(thrust::device_ptr<double> hK0,
                    thrust::device_ptr<double> hR0,
                    size_t len,
                    StreamPool& pool);
#endif

// CPU filter (available in all builds; no-op when filter_strength == 0).
void filter_old(double* gK, double* gR, double* hK0, double* hR0, size_t len);
void filter(double* gK, double* gR, double* hK0, double* hR0, size_t len);
void filter_dR(double* hR0, size_t len);
void filter_dRK(double* hK0, double* hR0, size_t len);
