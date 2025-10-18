#include "io/io_utils.hpp"
#include "simulation/simulation_data.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#if defined(H5_RUNTIME_OPTIONAL)
#include "io/h5_runtime.hpp"
#endif
#include <thread>
#include <mutex>
#include <condition_variable>

extern SimulationConfig config;
extern SimulationData* sim;
extern std::mutex saveMutex;
extern bool saveInProgress;
extern std::condition_variable saveCondition;

SimulationDataSnapshot saveSimulationState(const std::string& filename, double delta, double delta_t)
{
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));

    {
        std::unique_lock<std::mutex> lock(saveMutex);
        saveCondition.wait(lock, []{ return !saveInProgress; });
        saveInProgress = true;
    }

    _setSaveStart(filename);

#if DMFE_WITH_CUDA
    SimulationDataSnapshot snapshot = createDataSnapshot();
#else
    SimulationDataSnapshot snapshot;
    std::cerr << dmfe::console::WARN() << "Snapshot creation not available in CPU-only builds" << std::endl;
#endif

    if (config.async_export) {
        auto saveAsync = [filename, dirPath, delta, delta_t, snapshot]() {
            try {
#if defined(H5_RUNTIME_OPTIONAL)
                if (h5rt::available()) {
                    saveSimulationStateHDF5Async(filename, snapshot);
                    saveCompressedDataAsync(dirPath, snapshot);
                } else {
                    saveSimulationStateBinary(dirPath + "/data.bin", delta, delta_t);
                    saveCompressedDataAsync(dirPath, snapshot);
                }
#elif defined(USE_HDF5)
                saveSimulationStateHDF5Async(filename, snapshot);
                saveCompressedDataAsync(dirPath, snapshot);
#else
                saveSimulationStateBinary(dirPath + "/data.bin", delta, delta_t);
                saveCompressedDataAsync(dirPath, snapshot);
#endif
            } catch (const std::exception& e) {
                std::cerr << dmfe::console::ERR() << "Error in background save: " << e.what() << std::endl << std::flush;
                try {
                    saveSimulationStateBinary(dirPath + "/data.bin", delta, delta_t);
                    saveCompressedDataAsync(dirPath, snapshot);
                } catch (...) {
                    std::cerr << dmfe::console::ERR() << "Fallback save also failed" << std::endl << std::flush;
                }
            }
            {
                std::lock_guard<std::mutex> lock(saveMutex);
                saveInProgress = false;
            }
            _setSaveEnd(filename);
            saveCondition.notify_one();
        };
        std::thread saveThread(saveAsync);
        saveThread.detach();
    } else {
        try {
#if defined(H5_RUNTIME_OPTIONAL)
            if (h5rt::available()) {
                saveSimulationStateHDF5Async(filename, snapshot);
                saveCompressedDataAsync(dirPath, snapshot);
            } else {
                saveSimulationStateBinary(dirPath + "/data.bin", delta, delta_t);
                saveCompressedDataAsync(dirPath, snapshot);
            }
#elif defined(USE_HDF5)
            saveSimulationStateHDF5Async(filename, snapshot);
            saveCompressedDataAsync(dirPath, snapshot);
#else
            saveSimulationStateBinary(dirPath + "/data.bin", delta, delta_t);
            saveCompressedDataAsync(dirPath, snapshot);
#endif
        } catch (const std::exception& e) {
            std::cerr << dmfe::console::ERR() << "Error in synchronous save: " << e.what() << std::endl << std::flush;
            try {
                saveSimulationStateBinary(dirPath + "/data.bin", delta, delta_t);
                saveCompressedDataAsync(dirPath, snapshot);
            } catch (...) {
                std::cerr << dmfe::console::ERR() << "Fallback save also failed" << std::endl << std::flush;
            }
        }
        {
            std::lock_guard<std::mutex> lock(saveMutex);
            saveInProgress = false;
        }
        _setSaveEnd(filename);
        saveCondition.notify_one();
    }

    return snapshot;
}
