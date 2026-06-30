#pragma once

#include "host_simulation_data.hpp"

struct DeviceSimulationData;   // forward declaration

struct SimulationData {
    HostSimulationData* host = nullptr;
#if DMFE_WITH_CUDA
    DeviceSimulationData* device = nullptr;
#endif
};