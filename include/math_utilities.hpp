#ifndef MATH_UTILITIES_HPP
#define MATH_UTILITIES_HPP

#include <thrust/tuple.h>
#include <cmath>

// Lambda function for computing absolute difference between tuple elements
auto absDiff = [] __host__ __device__ (const thrust::tuple<double, double>& tup) {
    return fabs(thrust::get<0>(tup) - thrust::get<1>(tup));
};

// Functor for computing absolute difference between tuple elements
struct AbsDiff {
    __host__ __device__
    double operator()(const thrust::tuple<double, double>& tup) const {
        return fabs(thrust::get<0>(tup) - thrust::get<1>(tup));
    }
};

#endif // MATH_UTILITIES_HPP
