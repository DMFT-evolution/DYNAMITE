#include "simulation/device_simulation_data.hpp"
#include "EOMs/device_rk_data.hpp"

DeviceSimulationData* createDeviceSimulationData()
{
    return new DeviceSimulationData();
}

void destroyDeviceSimulationData(DeviceSimulationData* p)
{
    delete p;
}

DeviceRKData* createDeviceRKData()
{
    return new DeviceRKData();
}

void destroyDeviceRKData(DeviceRKData* p)
{
    delete p;
}