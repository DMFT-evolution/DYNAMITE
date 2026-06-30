#pragma once

#include <vector>
#include <cstddef>

struct HostRKData {
    size_t stages = 10;
    size_t posCount = 3;
    int init = 0;

    double *avec = nullptr;
    double *bvec = nullptr;
    double *b2vec = nullptr;
    double *cvec = nullptr;

    double gt = 0.0;
    double gtfinal = 0.0;
    double gte = 0.0;
    double ht = 0.0;
    double gt0 = 0.0;

    std::vector<double> gK, gR, gRfinal, gKfinal;
    std::vector<double> gKe, gRe, gK0, gR0;

    std::vector<double> posB1xvec, posB2xvec;

    std::vector<double> hK, hR, hK0, hR0;

    std::vector<double> d_avec;
};