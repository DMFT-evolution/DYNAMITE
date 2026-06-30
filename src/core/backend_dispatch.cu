// #include "EOMs/time_steps.hpp"

#include "EOMs/time_steps_cuda.cuh"
#include "EOMs/runge_kutta.cuh"
#include "core/device_utils.cuh"

#include "core/config.hpp"
#include "core/backend_dispatch.hpp"
#include "simulation/simulation_data.hpp"

extern SimulationConfig config;
extern SimulationData* sim;

double energy()
{
    if (config.gpu)
        return energyGPU(*sim->device,
                         config.T0);
    else
        return energyCPU();
}

double update(StreamPool* pool)
{
    if (config.gpu)
        return updateGPU(pool);
    else
        return updateCPU();
}


bool initializeGPUBackend(std::string& errorMessage)
{
    if (!isCompatibleGPUInstalled()) {
        errorMessage = "no compatible GPU found";
        return false;
    }

    if (!canCreateCudaStream(&errorMessage))
        return false;

    return true;
}

void synchronizeCompressedData()
{
    if (!config.gpu)
        return;

    auto copy = [](auto& host, const auto& device)
    {
        host.resize(device.size());
        thrust::copy(device.begin(), device.end(), host.begin());
    };

    copy(sim->host->QKB1int, sim->device->QKB1int);
    copy(sim->host->QRB1int, sim->device->QRB1int);
    copy(sim->host->theta,   sim->device->theta);
    copy(sim->host->t1grid,  sim->device->t1grid);
}