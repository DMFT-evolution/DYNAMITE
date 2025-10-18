#include "io/io_utils.hpp"
#include "simulation/simulation_data.hpp"
#include "core/gpu_memory_utils.hpp"
#include "math/math_ops.hpp"
#include "math/math_sigma.hpp"
#include "core/config.hpp"
#include "convolution/convolution.hpp"
#include "core/console.hpp"
#include "version/version_info.hpp"
#include "core/globals.hpp"           // globals if needed
#include "EOMs/time_steps.hpp"        // getLastLenEntries
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>

#if defined(H5_RUNTIME_OPTIONAL)
#include "io/h5_runtime.hpp"          // h5rt::available and helpers
#elif defined(USE_HDF5)
#include "H5Cpp.h"
#endif

using namespace std;

extern SimulationConfig config;
extern SimulationData* sim;

#if defined(H5_RUNTIME_OPTIONAL) || defined(USE_HDF5)
void saveSimulationStateHDF5(const std::string& filename, double delta, double delta_t)
{
#if defined(H5_RUNTIME_OPTIONAL)
    if (!h5rt::available()) throw std::runtime_error("HDF5 not available at runtime");
#if DMFE_WITH_CUDA
    if (config.gpu) {
        copyVectorsToCPU(*sim);
    }
#endif
    double energy;
    {
        vector<double> temp(config.len, 0.0);
        vector<double> lastQKv = getLastLenEntries(sim->h_QKv, config.len);
        SigmaK(lastQKv, temp);
        energy = -(ConvA(temp, getLastLenEntries(sim->h_QRv, config.len), sim->h_t1grid.back())[0]
                 + Dflambda(lastQKv[0]) / config.T0);
    }
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    std::string binFilename = dirPath + "/data.bin";
    if (fileExists(binFilename)) { std::remove(binFilename.c_str()); }

    auto file = h5rt::create_file_trunc(filename.c_str());
    if (file < 0) throw std::runtime_error("Failed to create HDF5 file");

    auto fail_and_fallback = [&](const char* why){
        std::cerr << dmfe::console::WARN() << "[HDF5] write failed: " << why << "; falling back to binary." << std::endl;
        h5rt::close_file(file);
        std::remove(filename.c_str());
        saveSimulationStateBinary(dirPath + "/data.bin", delta, delta_t);
    };

    const double p_start = 0.10, p_end = 0.50, p_span = (p_end - p_start);
    const double last_t1 = sim->h_t1grid.empty() ? 0.0 : sim->h_t1grid.back();
    auto update_prog = [&](size_t done, size_t total){
        double frac = p_start + (total ? (p_span * (double)done / (double)total) : 0.0);
        if (frac > p_end) frac = p_end; if (frac < p_start) frac = p_start;
        _setSaveProgress(frac, last_t1, "hdf5");
    };
    size_t total_elems = sim->h_QKv.size() + sim->h_QRv.size() + sim->h_dQKv.size() + sim->h_dQRv.size()
                       + sim->h_t1grid.size() + sim->h_rvec.size() + sim->h_drvec.size();
    size_t done_elems = 0; update_prog(done_elems, total_elems);

    if (!h5rt::write_dataset_1d_double(file, "QKv", sim->h_QKv.data(), sim->h_QKv.size())) { fail_and_fallback("QKv"); return; }
    done_elems += sim->h_QKv.size(); update_prog(done_elems, total_elems);
    if (!h5rt::write_dataset_1d_double(file, "QRv", sim->h_QRv.data(), sim->h_QRv.size())) { fail_and_fallback("QRv"); return; }
    done_elems += sim->h_QRv.size(); update_prog(done_elems, total_elems);
    if (!h5rt::write_dataset_1d_double(file, "dQKv", sim->h_dQKv.data(), sim->h_dQKv.size())) { fail_and_fallback("dQKv"); return; }
    done_elems += sim->h_dQKv.size(); update_prog(done_elems, total_elems);
    if (!h5rt::write_dataset_1d_double(file, "dQRv", sim->h_dQRv.data(), sim->h_dQRv.size())) { fail_and_fallback("dQRv"); return; }
    done_elems += sim->h_dQRv.size(); update_prog(done_elems, total_elems);
    if (!h5rt::write_dataset_1d_double(file, "t1grid", sim->h_t1grid.data(), sim->h_t1grid.size())) { fail_and_fallback("t1grid"); return; }
    done_elems += sim->h_t1grid.size(); update_prog(done_elems, total_elems);
    if (!h5rt::write_dataset_1d_double(file, "rvec", sim->h_rvec.data(), sim->h_rvec.size())) { fail_and_fallback("rvec"); return; }
    done_elems += sim->h_rvec.size(); update_prog(done_elems, total_elems);
    if (!h5rt::write_dataset_1d_double(file, "drvec", sim->h_drvec.data(), sim->h_drvec.size())) { fail_and_fallback("drvec"); return; }
    done_elems += sim->h_drvec.size(); update_prog(done_elems, total_elems);

    double t_current = sim->h_t1grid.back();
    int current_len = config.len; int current_loop = config.loop;
    if (!h5rt::write_attr_double(file, "time", t_current)) { fail_and_fallback("attr time"); return; }
    if (!h5rt::write_attr_int(file, "iteration", current_loop)) { fail_and_fallback("attr iteration"); return; }
    if (!h5rt::write_attr_int(file, "len", current_len)) { fail_and_fallback("attr len"); return; }
    if (!h5rt::write_attr_double(file, "delta", config.delta)) { fail_and_fallback("attr delta"); return; }
    if (!h5rt::write_attr_double(file, "delta_t", config.delta_t)) { fail_and_fallback("attr delta_t"); return; }
    if (!h5rt::write_attr_double(file, "T0", config.T0)) { fail_and_fallback("attr T0"); return; }
    if (!h5rt::write_attr_double(file, "lambda", config.lambda)) { fail_and_fallback("attr lambda"); return; }
    if (!h5rt::write_attr_int(file, "p", config.p)) { fail_and_fallback("attr p"); return; }
    if (!h5rt::write_attr_int(file, "p2", config.p2)) { fail_and_fallback("attr p2"); return; }
    if (!h5rt::write_attr_double(file, "Gamma", config.Gamma)) { fail_and_fallback("attr Gamma"); return; }
    if (!h5rt::write_attr_double(file, "delta_t_min", config.delta_t_min)) { fail_and_fallback("attr delta_t_min"); return; }
    if (!h5rt::write_attr_double(file, "delta_max", config.delta_max)) { fail_and_fallback("attr delta_max"); return; }
    if (!h5rt::write_attr_int(file, "use_serk2", config.use_serk2 ? 1 : 0)) { fail_and_fallback("attr use_serk2"); return; }
    if (!h5rt::write_attr_double(file, "energy", energy)) { fail_and_fallback("attr energy"); return; }

    h5rt::close_file(file);
    update_prog(total_elems, total_elems);
    if (config.debug) {
        dmfe::console::end_status_line_if_needed(dmfe::console::stdout_is_tty());
        invalidateStatusAnchor();
        std::cout << dmfe::console::SAVE() << "Saved HDF5 data to " << filename << std::endl;
    }
    saveParametersToFile(dirPath, delta, delta_t);
#if DMFE_WITH_CUDA
    saveHistory(filename, delta, delta_t, *sim, config.len, config.T0, config.gpu);
#endif
#elif defined(USE_HDF5)
#if DMFE_WITH_CUDA
    copyVectorsToCPU(*sim);
#endif
    double energy;
    {
        vector<double> temp(config.len, 0.0);
        vector<double> lastQKv = getLastLenEntries(sim->h_QKv, config.len);
        SigmaK(lastQKv, temp);
        energy = -(ConvA(temp, getLastLenEntries(sim->h_QRv, config.len), sim->h_t1grid.back())[0]
                 + Dflambda(lastQKv[0]) / config.T0);
    }
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    std::string binFilename = dirPath + "/data.bin";
    if (fileExists(binFilename)) { std::remove(binFilename.c_str()); }

    const double p_start = 0.10, p_end = 0.50, p_span = (p_end - p_start);
    const double last_t1 = sim->h_t1grid.empty() ? 0.0 : sim->h_t1grid.back();
    auto update_prog = [&](size_t done, size_t total){
        double frac = p_start + (p_span * (total ? (double)done / (double)total : 0.0));
        if (frac > p_end) frac = p_end; if (frac < p_start) frac = p_start; _setSaveProgress(frac, last_t1, "hdf5"); };
    size_t total_elems = sim->h_QKv.size() + sim->h_QRv.size() + sim->h_dQKv.size() + sim->h_dQRv.size()
                       + sim->h_t1grid.size() + sim->h_rvec.size() + sim->h_drvec.size();
    size_t done_elems = 0; update_prog(done_elems, total_elems);

    {
        H5::H5File file(filename, H5F_ACC_TRUNC);
        H5::DSetCreatPropList plist; plist.setDeflate(6); plist.setShuffle();
        {
            hsize_t dims[1] = {sim->h_QKv.size()}; H5::DataSpace sp(1, dims);
            size_t cs = std::min<size_t>(1048576, sim->h_QKv.size()); hsize_t cd[1] = {cs}; plist.setChunk(1, cd);
            file.createDataSet("QKv", H5::PredType::NATIVE_DOUBLE, sp, plist).write(sim->h_QKv.data(), H5::PredType::NATIVE_DOUBLE);
            done_elems += sim->h_QKv.size(); update_prog(done_elems, total_elems);
        }
        {
            hsize_t dims[1] = {sim->h_QRv.size()}; H5::DataSpace sp(1, dims);
            size_t cs = std::min<size_t>(1048576, sim->h_QRv.size()); hsize_t cd[1] = {cs}; plist.setChunk(1, cd);
            file.createDataSet("QRv", H5::PredType::NATIVE_DOUBLE, sp, plist).write(sim->h_QRv.data(), H5::PredType::NATIVE_DOUBLE);
            done_elems += sim->h_QRv.size(); update_prog(done_elems, total_elems);
        }
        {
            hsize_t dims[1] = {sim->h_dQKv.size()}; H5::DataSpace sp(1, dims);
            size_t cs = std::min<size_t>(1048576, sim->h_dQKv.size()); hsize_t cd[1] = {cs}; plist.setChunk(1, cd);
            file.createDataSet("dQKv", H5::PredType::NATIVE_DOUBLE, sp, plist).write(sim->h_dQKv.data(), H5::PredType::NATIVE_DOUBLE);
            done_elems += sim->h_dQKv.size(); update_prog(done_elems, total_elems);
        }
        {
            hsize_t dims[1] = {sim->h_dQRv.size()}; H5::DataSpace sp(1, dims);
            size_t cs = std::min<size_t>(1048576, sim->h_dQRv.size()); hsize_t cd[1] = {cs}; plist.setChunk(1, cd);
            file.createDataSet("dQRv", H5::PredType::NATIVE_DOUBLE, sp, plist).write(sim->h_dQRv.data(), H5::PredType::NATIVE_DOUBLE);
            done_elems += sim->h_dQRv.size(); update_prog(done_elems, total_elems);
        }
        {
            hsize_t dims[1] = {sim->h_t1grid.size()}; H5::DataSpace sp(1, dims);
            size_t cs = std::min<size_t>(1024, sim->h_t1grid.size()); hsize_t cd[1] = {cs}; plist.setChunk(1, cd);
            file.createDataSet("t1grid", H5::PredType::NATIVE_DOUBLE, sp, plist).write(sim->h_t1grid.data(), H5::PredType::NATIVE_DOUBLE);
            done_elems += sim->h_t1grid.size(); update_prog(done_elems, total_elems);
        }
        {
            hsize_t dims[1] = {sim->h_rvec.size()}; H5::DataSpace sp(1, dims);
            size_t cs = std::min<size_t>(1024, sim->h_rvec.size()); hsize_t cd[1] = {cs}; plist.setChunk(1, cd);
            file.createDataSet("rvec", H5::PredType::NATIVE_DOUBLE, sp, plist).write(sim->h_rvec.data(), H5::PredType::NATIVE_DOUBLE);
            done_elems += sim->h_rvec.size(); update_prog(done_elems, total_elems);
        }
        {
            hsize_t dims[1] = {sim->h_drvec.size()}; H5::DataSpace sp(1, dims);
            size_t cs = std::min<size_t>(1024, sim->h_drvec.size()); hsize_t cd[1] = {cs}; plist.setChunk(1, cd);
            file.createDataSet("drvec", H5::PredType::NATIVE_DOUBLE, sp, plist).write(sim->h_drvec.data(), H5::PredType::NATIVE_DOUBLE);
            done_elems += sim->h_drvec.size(); update_prog(done_elems, total_elems);
        }

        // Attributes
        H5::DataSpace scalar(H5S_SCALAR);
        auto add_attr = [&](const char* name, const H5::PredType& type, const void* value){ file.createAttribute(name, type, scalar).write(type, value); };
        double t_current = sim->h_t1grid.back(); int current_len = config.len; int current_loop = config.loop;
        add_attr("time", H5::PredType::NATIVE_DOUBLE, &t_current);
        add_attr("iteration", H5::PredType::NATIVE_INT, &current_loop);
        add_attr("len", H5::PredType::NATIVE_INT, &current_len);
        add_attr("delta", H5::PredType::NATIVE_DOUBLE, &config.delta);
        add_attr("delta_t", H5::PredType::NATIVE_DOUBLE, &config.delta_t);
        add_attr("T0", H5::PredType::NATIVE_DOUBLE, &config.T0);
        add_attr("lambda", H5::PredType::NATIVE_DOUBLE, &config.lambda);
        add_attr("p", H5::PredType::NATIVE_INT, &config.p);
        add_attr("p2", H5::PredType::NATIVE_INT, &config.p2);
        add_attr("Gamma", H5::PredType::NATIVE_DOUBLE, &config.Gamma);
        add_attr("delta_t_min", H5::PredType::NATIVE_DOUBLE, &config.delta_t_min);
        add_attr("delta_max", H5::PredType::NATIVE_DOUBLE, &config.delta_max);
    int use_serk2_int = config.use_serk2 ? 1 : 0; add_attr("use_serk2", H5::PredType::NATIVE_INT, &use_serk2_int);
        add_attr("energy", H5::PredType::NATIVE_DOUBLE, &energy);
    }
    update_prog(total_elems, total_elems);
    saveParametersToFile(dirPath, delta, delta_t);
#if DMFE_WITH_CUDA
    saveHistory(filename, delta, delta_t, *sim, config.len, config.T0, config.gpu);
#endif
#endif // runtime optional vs compile-time HDF5
}

void saveSimulationStateHDF5Async(const std::string& filename, const SimulationDataSnapshot& snapshot)
{
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    std::string binFilename = dirPath + "/data.bin";
    if (fileExists(binFilename)) { std::remove(binFilename.c_str()); }

#if defined(H5_RUNTIME_OPTIONAL)
    if (!h5rt::available()) throw std::runtime_error("HDF5 not available at runtime");
    auto file = h5rt::create_file_trunc(filename.c_str());
    if (file < 0) throw std::runtime_error("Failed to create HDF5 file");
    auto fail_and_fallback = [&](const char* why){
        std::cerr << dmfe::console::WARN() << "[HDF5] write failed: " << why << "; falling back to binary." << std::endl;
        h5rt::close_file(file);
        std::remove(filename.c_str());
        throw std::runtime_error(std::string("HDF5 write failed: ") + why);
    };
    const double p_start = 0.10, p_end = 0.50, p_span = (p_end - p_start);
    const double last_t1 = snapshot.t1grid.empty() ? 0.0 : snapshot.t1grid.back();
    auto update_prog = [&](size_t done, size_t total){ double frac = p_start + (total ? (p_span * (double)done / (double)total) : 0.0); if (frac > p_end) frac = p_end; if (frac < p_start) frac = p_start; _setSaveProgress(frac, last_t1, "hdf5"); };
    size_t total_elems = snapshot.QKv.size() + snapshot.QRv.size() + snapshot.dQKv.size() + snapshot.dQRv.size() + snapshot.t1grid.size() + snapshot.rvec.size() + snapshot.drvec.size();
    size_t done_elems = 0; update_prog(done_elems, total_elems);
    if (!h5rt::write_dataset_1d_double(file, "QKv", snapshot.QKv.data(), snapshot.QKv.size())) { fail_and_fallback("QKv"); }
    done_elems += snapshot.QKv.size(); update_prog(done_elems, total_elems);
    if (!h5rt::write_dataset_1d_double(file, "QRv", snapshot.QRv.data(), snapshot.QRv.size())) { fail_and_fallback("QRv"); }
    done_elems += snapshot.QRv.size(); update_prog(done_elems, total_elems);
    if (!h5rt::write_dataset_1d_double(file, "dQKv", snapshot.dQKv.data(), snapshot.dQKv.size())) { fail_and_fallback("dQKv"); }
    done_elems += snapshot.dQKv.size(); update_prog(done_elems, total_elems);
    if (!h5rt::write_dataset_1d_double(file, "dQRv", snapshot.dQRv.data(), snapshot.dQRv.size())) { fail_and_fallback("dQRv"); }
    done_elems += snapshot.dQRv.size(); update_prog(done_elems, total_elems);
    if (!h5rt::write_dataset_1d_double(file, "t1grid", snapshot.t1grid.data(), snapshot.t1grid.size())) { fail_and_fallback("t1grid"); }
    done_elems += snapshot.t1grid.size(); update_prog(done_elems, total_elems);
    if (!h5rt::write_dataset_1d_double(file, "rvec", snapshot.rvec.data(), snapshot.rvec.size())) { fail_and_fallback("rvec"); }
    done_elems += snapshot.rvec.size(); update_prog(done_elems, total_elems);
    if (!h5rt::write_dataset_1d_double(file, "drvec", snapshot.drvec.data(), snapshot.drvec.size())) { fail_and_fallback("drvec"); }
    done_elems += snapshot.drvec.size(); update_prog(done_elems, total_elems);
    if (!h5rt::write_attr_double(file, "time", snapshot.t_current)) { fail_and_fallback("attr time"); }
    if (!h5rt::write_attr_int(file, "iteration", snapshot.current_loop)) { fail_and_fallback("attr iteration"); }
    if (!h5rt::write_attr_int(file, "len", snapshot.current_len)) { fail_and_fallback("attr len"); }
    if (!h5rt::write_attr_double(file, "delta", snapshot.config_snapshot.delta)) { fail_and_fallback("attr delta"); }
    if (!h5rt::write_attr_double(file, "delta_t", snapshot.config_snapshot.delta_t)) { fail_and_fallback("attr delta_t"); }
    if (!h5rt::write_attr_double(file, "T0", snapshot.config_snapshot.T0)) { fail_and_fallback("attr T0"); }
    if (!h5rt::write_attr_double(file, "lambda", snapshot.config_snapshot.lambda)) { fail_and_fallback("attr lambda"); }
    if (!h5rt::write_attr_int(file, "p", snapshot.config_snapshot.p)) { fail_and_fallback("attr p"); }
    if (!h5rt::write_attr_int(file, "p2", snapshot.config_snapshot.p2)) { fail_and_fallback("attr p2"); }
    if (!h5rt::write_attr_double(file, "Gamma", snapshot.config_snapshot.Gamma)) { fail_and_fallback("attr Gamma"); }
    if (!h5rt::write_attr_double(file, "delta_t_min", snapshot.config_snapshot.delta_t_min)) { fail_and_fallback("attr delta_t_min"); }
    if (!h5rt::write_attr_double(file, "delta_max", snapshot.config_snapshot.delta_max)) { fail_and_fallback("attr delta_max"); }
    if (!h5rt::write_attr_int(file, "use_serk2", snapshot.config_snapshot.use_serk2 ? 1 : 0)) { fail_and_fallback("attr use_serk2"); }
    if (!h5rt::write_attr_double(file, "energy", snapshot.energy)) { fail_and_fallback("attr energy"); }
    h5rt::close_file(file);
    update_prog(total_elems, total_elems);
    if (config.debug) {
        std::cout << dmfe::console::SAVE() << "Saved HDF5 data to " << filename << " (async)" << std::endl;
    }
    saveParametersToFileAsync(dirPath, snapshot.config_snapshot.delta, snapshot.config_snapshot.delta_t, snapshot);
#if DMFE_WITH_CUDA
    saveHistoryAsync(filename, snapshot.config_snapshot.delta, snapshot.config_snapshot.delta_t, snapshot);
#endif
#elif defined(USE_HDF5)
    const double p_start = 0.10, p_end = 0.50, p_span = (p_end - p_start);
    const double last_t1 = snapshot.t1grid.empty() ? 0.0 : snapshot.t1grid.back();
    auto update_prog = [&](size_t done, size_t total){ double frac = p_start + (total ? (p_span * (double)done / (double)total) : 0.0); if (frac > p_end) frac = p_end; if (frac < p_start) frac = p_start; _setSaveProgress(frac, last_t1, "hdf5"); };
    size_t total_elems = snapshot.QKv.size() + snapshot.QRv.size() + snapshot.dQKv.size() + snapshot.dQRv.size() + snapshot.t1grid.size() + snapshot.rvec.size() + snapshot.drvec.size();
    size_t done_elems = 0; update_prog(done_elems, total_elems);

    {
        H5::H5File file(filename, H5F_ACC_TRUNC);
        H5::DSetCreatPropList plist; plist.setDeflate(6); plist.setShuffle();
        auto write = [&](const char* name, const std::vector<double>& v){ hsize_t d[1] = {v.size()}; H5::DataSpace sp(1, d); size_t cs = std::min<size_t>(1048576, v.size()); hsize_t cd[1] = {cs}; plist.setChunk(1, cd); file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, sp, plist).write(v.data(), H5::PredType::NATIVE_DOUBLE); done_elems += v.size(); update_prog(done_elems, total_elems); };
        write("QKv", snapshot.QKv); write("QRv", snapshot.QRv); write("dQKv", snapshot.dQKv); write("dQRv", snapshot.dQRv);
        auto write_small = [&](const char* name, const std::vector<double>& v){ hsize_t d[1] = {v.size()}; H5::DataSpace sp(1, d); size_t cs = std::min<size_t>(1024, v.size()); hsize_t cd[1] = {cs}; plist.setChunk(1, cd); file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, sp, plist).write(v.data(), H5::PredType::NATIVE_DOUBLE); done_elems += v.size(); update_prog(done_elems, total_elems); };
        write_small("t1grid", snapshot.t1grid); write_small("rvec", snapshot.rvec); write_small("drvec", snapshot.drvec);

        H5::DataSpace scalar(H5S_SCALAR);
        auto add_attr = [&](const char* name, const H5::PredType& type, const void* value){ file.createAttribute(name, type, scalar).write(type, value); };
        add_attr("time", H5::PredType::NATIVE_DOUBLE, &snapshot.t_current);
        add_attr("iteration", H5::PredType::NATIVE_INT, &snapshot.current_loop);
        add_attr("len", H5::PredType::NATIVE_INT, &snapshot.current_len);
        add_attr("delta", H5::PredType::NATIVE_DOUBLE, &snapshot.config_snapshot.delta);
        add_attr("delta_t", H5::PredType::NATIVE_DOUBLE, &snapshot.config_snapshot.delta_t);
        add_attr("T0", H5::PredType::NATIVE_DOUBLE, &snapshot.config_snapshot.T0);
        add_attr("lambda", H5::PredType::NATIVE_DOUBLE, &snapshot.config_snapshot.lambda);
        add_attr("p", H5::PredType::NATIVE_INT, &snapshot.config_snapshot.p);
        add_attr("p2", H5::PredType::NATIVE_INT, &snapshot.config_snapshot.p2);
        add_attr("Gamma", H5::PredType::NATIVE_DOUBLE, &snapshot.config_snapshot.Gamma);
        add_attr("delta_t_min", H5::PredType::NATIVE_DOUBLE, &snapshot.config_snapshot.delta_t_min);
        add_attr("delta_max", H5::PredType::NATIVE_DOUBLE, &snapshot.config_snapshot.delta_max);
    int use_serk2_int = snapshot.config_snapshot.use_serk2 ? 1 : 0; add_attr("use_serk2", H5::PredType::NATIVE_INT, &use_serk2_int);
    // aggressive_sparsify attribute removed; not written anymore
    add_attr("energy", H5::PredType::NATIVE_DOUBLE, &snapshot.energy);
    }
    update_prog(total_elems, total_elems);
    saveParametersToFileAsync(dirPath, snapshot.config_snapshot.delta, snapshot.config_snapshot.delta_t, snapshot);
#if DMFE_WITH_CUDA
    saveHistoryAsync(filename, snapshot.config_snapshot.delta, snapshot.config_snapshot.delta_t, snapshot);
#endif
#endif
}
#endif // defined(H5_RUNTIME_OPTIONAL) || defined(USE_HDF5)
