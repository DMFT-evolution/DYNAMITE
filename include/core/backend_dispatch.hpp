#pragma once

#include <string>

class StreamPool;

double update(StreamPool* pool);

double energy();

bool initializeGPUBackend(std::string& errorMessage);

void synchronizeCompressedData();