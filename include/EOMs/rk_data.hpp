#pragma once

#include "EOMs/host_rk_data.hpp"

struct DeviceRKData;

struct RKData
{
    HostRKData* host = nullptr;
    DeviceRKData* device = nullptr;
};