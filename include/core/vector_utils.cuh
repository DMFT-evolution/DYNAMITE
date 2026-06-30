#pragma once
#include "core/vector_utils.hpp"

#if DMFE_WITH_CUDA
#include "core/device_vector.hpp"
#include <thrust/device_ptr.h>

// Slice helpers

template <typename T>
dmfe::device_vector<T> get_slice(const dmfe::device_vector<T>& vec, size_t i, size_t len) {
    size_t start = i * len;
    size_t end = start + len;
    if (end > vec.size()) throw std::out_of_range("get_slice: slice out of range");
    return dmfe::device_vector<T>(vec.begin() + start, vec.begin() + end);
}

template <typename T>
thrust::device_ptr<T> get_slice_ptr(const dmfe::device_vector<T>& vec, size_t i, size_t len) {
    size_t start = i * len;
    if (start + len > vec.size()) throw std::out_of_range("get_slice_ptr: slice out of range");
    return thrust::device_ptr<T>(const_cast<T*>(thrust::raw_pointer_cast(vec.data()) + start));
}

template <typename T>
void set_slice(dmfe::device_vector<T>& vec, size_t i, const dmfe::device_vector<T>& slice) {
    size_t start = i * slice.size();
    size_t end = start + slice.size();
    if (end > vec.size()) throw std::out_of_range("set_slice: slice write out of range");
    thrust::copy(slice.begin(), slice.end(), vec.begin() + start);
}

template <typename T>
void set_slice_ptr(dmfe::device_vector<T>& vec, size_t i, const thrust::device_ptr<T>& slice, size_t slice_len) {
    size_t start = i * slice_len;
    size_t end = start + slice_len;
    if (end > vec.size()) throw std::out_of_range("set_slice_ptr: slice write out of range");
    thrust::copy(slice, slice + slice_len, vec.begin() + start);
}
#endif // DMFE_WITH_CUDA