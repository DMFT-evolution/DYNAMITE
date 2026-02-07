#include "core/stream_pool.hpp"
#include "core/console.hpp"
#include <iostream>
#include <stdexcept>

StreamPool::StreamPool(size_t N) {
    (void)cudaGetLastError();
    streams.resize(N);
    for (auto &s : streams) {
        cudaError_t err = cudaStreamCreate(&s);
        if (err != cudaSuccess) {
            std::cerr << dmfe::console::WARN()
                      << "Failed to create CUDA stream (" << cudaGetErrorString(err)
                      << "). Falling back to default stream." << std::endl;
            s = 0;
        }
    }
}

StreamPool::~StreamPool() {
    for (auto &s : streams) {
        if (s != 0) {
            cudaStreamDestroy(s);
        }
    }
}

StreamPool& getDefaultStreamPool() {
    static StreamPool defaultPool(20);
    return defaultPool;
}
