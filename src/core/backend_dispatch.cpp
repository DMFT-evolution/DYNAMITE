#include "core/backend_dispatch.hpp"

#include "EOMs/time_steps.hpp"
#include "EOMs/runge_kutta.hpp"

double energy()
{
    return energyCPU();
}

double update(StreamPool*)
{
    return updateCPU();
}

bool initializeGPUBackend(std::string& errorMessage)
{
    errorMessage = "program was built without CUDA support";
    return false;
}

void synchronizeCompressedData()
{
    // host already synchronized
}