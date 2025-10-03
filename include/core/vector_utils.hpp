#pragma once
#include "core/config_build.hpp"
#include <vector>
#include <stdexcept>
#include <algorithm>

#if DMFE_WITH_CUDA
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

// Slice helpers

template <typename T>
thrust::device_vector<T> get_slice(const thrust::device_vector<T>& vec, size_t i, size_t len) {
    size_t start = i * len;
    size_t end = start + len;
    if (end > vec.size()) throw std::out_of_range("get_slice: slice out of range");
    return thrust::device_vector<T>(vec.begin() + start, vec.begin() + end);
}

template <typename T>
thrust::device_ptr<T> get_slice_ptr(const thrust::device_vector<T>& vec, size_t i, size_t len) {
    size_t start = i * len;
    if (start + len > vec.size()) throw std::out_of_range("get_slice_ptr: slice out of range");
    return thrust::device_ptr<T>(const_cast<T*>(thrust::raw_pointer_cast(vec.data()) + start));
}

template <typename T>
void set_slice(thrust::device_vector<T>& vec, size_t i, const thrust::device_vector<T>& slice) {
    size_t start = i * slice.size();
    size_t end = start + slice.size();
    if (end > vec.size()) throw std::out_of_range("set_slice: slice write out of range");
    thrust::copy(slice.begin(), slice.end(), vec.begin() + start);
}

template <typename T>
void set_slice_ptr(thrust::device_vector<T>& vec, size_t i, const thrust::device_ptr<T>& slice, size_t slice_len) {
    size_t start = i * slice_len;
    size_t end = start + slice_len;
    if (end > vec.size()) throw std::out_of_range("set_slice_ptr: slice write out of range");
    thrust::copy(slice, slice + slice_len, vec.begin() + start);
}
#endif // DMFE_WITH_CUDA

// Host vector operators
inline std::vector<double>& operator+=(std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) throw std::invalid_argument("Vectors must be same size for +=");
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), std::plus<double>());
    return lhs;
}
inline std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors must match for +");
    std::vector<double> r(a.size());
    std::transform(a.begin(), a.end(), b.begin(), r.begin(), std::plus<>());
    return r;
}
inline std::vector<double> operator-(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors must match for -");
    std::vector<double> r(a.size());
    std::transform(a.begin(), a.end(), b.begin(), r.begin(), std::minus<>());
    return r;
}
inline std::vector<double> operator*(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors must match for * element-wise");
    std::vector<double> r(a.size());
    std::transform(a.begin(), a.end(), b.begin(), r.begin(), std::multiplies<>());
    return r;
}
inline std::vector<double> operator*(const std::vector<double>& a, double s) {
    std::vector<double> r(a.size());
    std::transform(a.begin(), a.end(), r.begin(), [s](double v){return v*s;});
    return r;
}
inline std::vector<double> operator*(double s, const std::vector<double>& a) { return a * s; }
