#pragma once
#include <vector>
#include "config.hpp"
#include <cuda_runtime.h>

class StreamPool {
    std::vector<cudaStream_t> streams;
public:
    explicit StreamPool(size_t N);
    ~StreamPool();
    cudaStream_t operator[](size_t i) const { return streams[i]; }
    size_t size() const { return streams.size(); }
};

StreamPool& getDefaultStreamPool();
