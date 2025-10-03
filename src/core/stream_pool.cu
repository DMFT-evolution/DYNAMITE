#include "core/stream_pool.hpp"
#include <stdexcept>

StreamPool::StreamPool(size_t N) {
    streams.resize(N);
    for (auto &s : streams) {
        cudaError_t err = cudaStreamCreate(&s);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream");
        }
    }
}

StreamPool::~StreamPool() {
    for (auto &s : streams) cudaStreamDestroy(s);
}

StreamPool& getDefaultStreamPool() {
    static StreamPool defaultPool(20);
    return defaultPool;
}
