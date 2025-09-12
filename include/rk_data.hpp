#pragma once
#include <thrust/device_vector.h>
#include <cstddef>

struct RKData {
    size_t stages = 10;
    size_t posCount = 3;
    int init = 0; // Flag to indicate which RK data is initialized
    double *avec = nullptr, *bvec = nullptr, *b2vec = nullptr, *cvec = nullptr;
    double gt = 0.0, gtfinal = 0.0, gte = 0.0, ht = 0.0, gt0 = 0.0;
    thrust::device_vector<double> gK, gR, gRfinal, gKfinal, gKe, gRe, gK0, gR0;
    thrust::device_vector<double> posB1xvec, posB2xvec;
    thrust::device_vector<double> hK, hR, hK0, hR0, d_avec;
};
