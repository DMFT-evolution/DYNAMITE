#include "io/io_utils.hpp"
#include "simulation/simulation_data.hpp"
#include "core/gpu_memory_utils.hpp"
#include "math/math_ops.hpp"
#include "math/math_sigma.hpp"
#include "core/config.hpp"
#include "convolution/convolution.hpp"
#include "core/globals.hpp"
           // globals
#include "EOMs/time_steps.hpp"        // getLastLenEntries
#include "core/console.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

extern SimulationConfig config;
extern SimulationData* sim;
// rk not used here; avoid requiring rk_data in this TU

void saveSimulationStateBinary(const std::string& filename, double delta, double delta_t)
{
#if DMFE_WITH_CUDA
    if (config.gpu) {
        copyVectorsToCPU(*sim);
    }
#endif

    // Calculate energy before saving (CPU path, using host arrays)
    double energy;
    {
        vector<double> temp(config.len, 0.0);
        vector<double> lastQKv = getLastLenEntries(sim->host->QKv, config.len);
        SigmaK(lastQKv, temp);
        energy = -(ConvA(temp, getLastLenEntries(sim->host->QRv, config.len), sim->host->t1grid.back())[0]
                 + Dflambda(lastQKv[0]) / config.T0);
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << dmfe::console::ERR() << "Could not open file " << filename << std::endl;
        return;
    }

    // Progress mapping for main file write: [0.10 .. 0.50]
    const double p_start = 0.10;
    const double p_end   = 0.50;
    const double p_span  = (p_end - p_start);
    const double last_t1 = sim->host->t1grid.empty() ? 0.0 : sim->host->t1grid.back();

    auto update_prog = [&](size_t done, size_t total){
        double frac = p_start + (total ? (p_span * (double)done / (double)total) : 0.0);
        if (frac > p_end) frac = p_end;
        if (frac < p_start) frac = p_start;
        _setSaveProgress(frac, last_t1, "binary");
    };

    size_t total_elems = sim->host->t1grid.size()
                       + sim->host->QKv.size() + sim->host->QRv.size()
                       + sim->host->dQKv.size() + sim->host->dQRv.size()
                       + sim->host->rvec.size() + sim->host->drvec.size();
    size_t done_elems = 0;
    update_prog(done_elems, total_elems);

    int header_version = 1;
    file.write(reinterpret_cast<char*>(&header_version), sizeof(int));

    size_t t1grid_size = sim->host->t1grid.size();
    size_t vector_len = config.len;
    file.write(reinterpret_cast<char*>(&t1grid_size), sizeof(size_t));
    file.write(reinterpret_cast<char*>(&vector_len), sizeof(size_t));

    // Write parameters
    file.write(reinterpret_cast<const char*>(&config.p), sizeof(int));
    file.write(reinterpret_cast<const char*>(&config.p2), sizeof(int));
    file.write(reinterpret_cast<const char*>(&config.lambda), sizeof(double));
    file.write(reinterpret_cast<const char*>(&config.T0), sizeof(double));
    file.write(reinterpret_cast<const char*>(&config.Gamma), sizeof(double));
    file.write(reinterpret_cast<const char*>(&config.delta), sizeof(double));
    file.write(reinterpret_cast<const char*>(&config.delta_t), sizeof(double));
    file.write(reinterpret_cast<const char*>(&config.loop), sizeof(int));
    file.write(reinterpret_cast<const char*>(&energy), sizeof(double));

    // Write arrays
    file.write(reinterpret_cast<const char*>(sim->host->t1grid.data()), sim->host->t1grid.size() * sizeof(double));
    done_elems += sim->host->t1grid.size(); update_prog(done_elems, total_elems);

    file.write(reinterpret_cast<const char*>(sim->host->QKv.data()), sim->host->QKv.size() * sizeof(double));
    done_elems += sim->host->QKv.size(); update_prog(done_elems, total_elems);
    file.write(reinterpret_cast<const char*>(sim->host->QRv.data()), sim->host->QRv.size() * sizeof(double));
    done_elems += sim->host->QRv.size(); update_prog(done_elems, total_elems);
    file.write(reinterpret_cast<const char*>(sim->host->dQKv.data()), sim->host->dQKv.size() * sizeof(double));
    done_elems += sim->host->dQKv.size(); update_prog(done_elems, total_elems);
    file.write(reinterpret_cast<const char*>(sim->host->dQRv.data()), sim->host->dQRv.size() * sizeof(double));
    done_elems += sim->host->dQRv.size(); update_prog(done_elems, total_elems);
    file.write(reinterpret_cast<const char*>(sim->host->rvec.data()), sim->host->rvec.size() * sizeof(double));
    done_elems += sim->host->rvec.size(); update_prog(done_elems, total_elems);
    file.write(reinterpret_cast<const char*>(sim->host->drvec.data()), sim->host->drvec.size() * sizeof(double));
    done_elems += sim->host->drvec.size(); update_prog(done_elems, total_elems);
    file.flush();
    update_prog(total_elems, total_elems);

    if (config.debug) {
        dmfe::console::end_status_line_if_needed(dmfe::console::stdout_is_tty());
        invalidateStatusAnchor();
        std::cout << dmfe::console::SAVE() << "Saved binary data to " << filename << std::endl;
    }

    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    saveParametersToFile(dirPath, delta, delta_t);

#if DMFE_WITH_CUDA
    saveHistory(filename, delta, delta_t, *sim, config.len, config.T0, config.gpu);
#endif
}
