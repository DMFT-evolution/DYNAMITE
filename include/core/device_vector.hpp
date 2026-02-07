#pragma once
#include "core/config_build.hpp"

#if DMFE_WITH_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

namespace dmfe {
namespace detail {
inline bool async_pool_supported() {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
    static int supported = -1;
    if (supported >= 0) {
        return supported != 0;
    }
    int device = 0;
    cudaError_t dev_err = cudaGetDevice(&device);
    if (dev_err != cudaSuccess) {
        supported = 0;
        return false;
    }
    int attr = 0;
    cudaError_t err = cudaDeviceGetAttribute(&attr, cudaDevAttrMemoryPoolsSupported, device);
    if (err != cudaSuccess) {
        supported = 0;
        return false;
    }
    supported = attr;
    return supported != 0;
#else
    return false;
#endif
}

inline void configure_async_pool_once() {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
    static bool configured = false;
    if (configured) {
        return;
    }
    configured = true;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return;
    }
    cudaMemPool_t pool = nullptr;
    if (cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess) {
        return;
    }
    std::uint64_t threshold = std::numeric_limits<std::uint64_t>::max();
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
#endif
}

inline void* allocate_device_bytes(std::size_t bytes, cudaStream_t stream) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
    if (stream != 0 && async_pool_supported()) {
        configure_async_pool_once();
        void* ptr = nullptr;
        cudaError_t err = cudaMallocAsync(&ptr, bytes, stream);
        if (err == cudaSuccess) {
            return ptr;
        }
        // Fall back if async allocation fails for any reason.
    }
#endif
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw thrust::system_error(err, thrust::cuda_category(), "cudaMalloc failed");
    }
    return ptr;
}

inline void deallocate_device_bytes(void* ptr, cudaStream_t stream) {
    if (!ptr) {
        return;
    }
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
    if (stream != 0 && async_pool_supported()) {
        cudaError_t err = cudaFreeAsync(ptr, stream);
        if (err == cudaSuccess) {
            return;
        }
        // Fall back if async free fails for any reason.
    }
#endif
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        throw thrust::system_error(err, thrust::cuda_category(), "cudaFree failed");
    }
}
} // namespace detail

template <typename T>
class cuda_async_allocator {
public:
    using value_type = T;
    using pointer = thrust::device_ptr<T>;
    using const_pointer = thrust::device_ptr<const T>;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    cuda_async_allocator() noexcept : stream_(0) {}
    explicit cuda_async_allocator(cudaStream_t stream) noexcept : stream_(stream) {}

    template <class U>
    cuda_async_allocator(const cuda_async_allocator<U>& other) noexcept : stream_(other.stream()) {}

    pointer allocate(std::size_t n) {
        if (n > max_size()) {
            throw std::bad_alloc();
        }
        std::size_t bytes = n * sizeof(T);
        return thrust::device_pointer_cast(static_cast<T*>(detail::allocate_device_bytes(bytes, stream_)));
    }

    void deallocate(pointer ptr, std::size_t) noexcept {
        try {
            detail::deallocate_device_bytes(ptr.get(), stream_);
        } catch (...) {
            // Thrust allocators are required not to throw from deallocate.
        }
    }

    std::size_t max_size() const noexcept {
        return std::numeric_limits<std::size_t>::max() / sizeof(T);
    }

    cudaStream_t stream() const noexcept { return stream_; }

    template <class U>
    struct rebind {
        using other = cuda_async_allocator<U>;
    };

    bool operator==(const cuda_async_allocator& other) const noexcept {
        return stream_ == other.stream_;
    }

    bool operator!=(const cuda_async_allocator& other) const noexcept {
        return !(*this == other);
    }

private:
    cudaStream_t stream_;
};

template <typename T>
using device_vector = thrust::device_vector<T, cuda_async_allocator<T>>;

} // namespace dmfe

#else
#include <vector>

namespace dmfe {
template <typename T>
using device_vector = std::vector<T>;
}

#endif
