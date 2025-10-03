#pragma once
#include "core/config_build.hpp"
#include <vector>
#include "core/config.hpp"

#if DMFE_WITH_CUDA
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
#endif
