#include "io/io_utils.hpp"
#include "simulation/simulation_data.hpp"
#include "core/gpu_memory_utils.hpp"
#include "math/math_ops.hpp"
#include "math/math_sigma.hpp"
#include "core/config.hpp"
#include "core/globals.hpp"
#include "convolution/convolution.hpp"
#include "core/device_utils.cuh"
#include "EOMs/time_steps.hpp"
#include "version/version_info.hpp"
#include <fstream>
#include <iostream>
#include "core/console.hpp"
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#if DMFE_WITH_CUDA
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#endif
#include <unistd.h>
#include <limits.h>
#include <errno.h>
#include <cstring>
#include <chrono>
#include <ctime>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

#if defined(H5_RUNTIME_OPTIONAL)
#include "io/h5_runtime.hpp"
#elif defined(USE_HDF5)
#include "H5Cpp.h"
#endif

#include "version/version_compat.hpp"

using namespace std;

// External global variables
extern SimulationConfig config;
extern SimulationData* sim;
extern RKData* rk;
extern size_t peak_memory_kb;
extern size_t peak_gpu_memory_mb;
extern std::chrono::high_resolution_clock::time_point program_start_time;

// Global variables for async save synchronization
extern std::mutex saveMutex;
extern bool saveInProgress;
extern std::condition_variable saveCondition;

// Basic imports
std::vector<double> importVectorFromFile(const std::string &filename)
{
    std::vector<double> data;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << dmfe::console::ERR() << "Could not open file " << filename << std::endl;
        return data;
    }
    double v;
    while (file >> v)
        data.push_back(v);
    return data;
}

std::vector<size_t> importIntVectorFromFile(const std::string &filename)
{
    std::vector<size_t> data;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << dmfe::console::ERR() << "Could not open file " << filename << std::endl;
        return data;
    }
    size_t v;
    while (file >> v)
        data.push_back(v);
    return data;
}

void import(SimulationData &sim, size_t len_param, int& ord_ref)
{
    std::ostringstream base;
    base << "Grid_data/" << len_param << "/";
    std::string prefix = base.str();
    sim.h_theta = importVectorFromFile(prefix + "theta.dat");
    sim.h_phi1 = importVectorFromFile(prefix + "phi1.dat");
    sim.h_phi2 = importVectorFromFile(prefix + "phi2.dat");
    sim.h_posA1y = importVectorFromFile(prefix + "posA1y.dat");
    sim.h_posA2y = importVectorFromFile(prefix + "posA2y.dat");
    sim.h_posB2y = importVectorFromFile(prefix + "posB2y.dat");
    sim.h_indsA1y = importIntVectorFromFile(prefix + "indsA1y.dat");
    sim.h_indsA2y = importIntVectorFromFile(prefix + "indsA2y.dat");
    sim.h_indsB2y = importIntVectorFromFile(prefix + "indsB2y.dat");
    sim.h_weightsA1y = importVectorFromFile(prefix + "weightsA1y.dat");
    sim.h_weightsA2y = importVectorFromFile(prefix + "weightsA2y.dat");
    sim.h_weightsB2y = importVectorFromFile(prefix + "weightsB2y.dat");
    sim.h_integ = importVectorFromFile(prefix + "int.dat");
    if (len_param)
        ord_ref = sim.h_weightsB2y.empty() ? 0 : static_cast<int>(sim.h_weightsB2y.size() / (len_param * len_param) - 2);
}

bool loadSimulationStateBinary(const std::string &filename, SimulationData &sim,
                              int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param,
                              size_t len_param, double delta_t_min_param, double delta_max_param,
                              bool use_serk2_param,
                              LoadedStateParams& loaded_params)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << dmfe::console::ERR() << "Could not open binary file " << filename << std::endl;
        return false;
    }
    int header_version;
    file.read(reinterpret_cast<char *>(&header_version), sizeof(int));
    if (header_version != 1)
    {
        std::cerr << dmfe::console::ERR() << "Unsupported binary file version: " << header_version << std::endl;
        return false;
    }
    size_t t1grid_size, vector_len;
    file.read(reinterpret_cast<char *>(&t1grid_size), sizeof(size_t));
    file.read(reinterpret_cast<char *>(&vector_len), sizeof(size_t));
    int file_p, file_p2;
    double file_lambda, file_T0, file_Gamma, file_delta, file_delta_t;
    int file_loop;
    double file_energy;
    file.read(reinterpret_cast<char *>(&file_p), sizeof(int));
    file.read(reinterpret_cast<char *>(&file_p2), sizeof(int));
    file.read(reinterpret_cast<char *>(&file_lambda), sizeof(double));
    file.read(reinterpret_cast<char *>(&file_T0), sizeof(double));
    file.read(reinterpret_cast<char *>(&file_Gamma), sizeof(double));
    file.read(reinterpret_cast<char *>(&file_delta), sizeof(double));
    file.read(reinterpret_cast<char *>(&file_delta_t), sizeof(double));
    file.read(reinterpret_cast<char *>(&file_loop), sizeof(int));
    file.read(reinterpret_cast<char *>(&file_energy), sizeof(double));
    if (file_p != p_param || file_p2 != p2_param || fabs(file_lambda - lambda_param) > 1e-10 || fabs(file_T0 - T0_param) > 1e-10 || fabs(file_Gamma - Gamma_param) > 1e-10)
    {
        std::cerr << dmfe::console::WARN() << "File parameters don't match current simulation parameters" << std::endl;
        return false;
    }
    sim.h_t1grid.resize(t1grid_size);
    file.read(reinterpret_cast<char *>(sim.h_t1grid.data()), t1grid_size * sizeof(double));
    sim.h_QKv.resize(t1grid_size * vector_len);
    file.read(reinterpret_cast<char *>(sim.h_QKv.data()), sim.h_QKv.size() * sizeof(double));
    sim.h_QRv.resize(t1grid_size * vector_len);
    file.read(reinterpret_cast<char *>(sim.h_QRv.data()), sim.h_QRv.size() * sizeof(double));
    sim.h_dQKv.resize(t1grid_size * vector_len);
    file.read(reinterpret_cast<char *>(sim.h_dQKv.data()), sim.h_dQKv.size() * sizeof(double));
    sim.h_dQRv.resize(t1grid_size * vector_len);
    file.read(reinterpret_cast<char *>(sim.h_dQRv.data()), sim.h_dQRv.size() * sizeof(double));
    sim.h_rvec.resize(t1grid_size);
    file.read(reinterpret_cast<char *>(sim.h_rvec.data()), sim.h_rvec.size() * sizeof(double));
    sim.h_drvec.resize(t1grid_size);
    file.read(reinterpret_cast<char *>(sim.h_drvec.data()), sim.h_drvec.size() * sizeof(double));
    loaded_params.delta = file_delta;
    loaded_params.delta_t = file_delta_t;
    loaded_params.loop = file_loop;
    sim.h_delta_t_ratio.resize(t1grid_size);
    sim.h_delta_t_ratio[0] = 0.0;
    for (size_t i = 2; i < t1grid_size; ++i)
        sim.h_delta_t_ratio[i] = (sim.h_t1grid[i] - sim.h_t1grid[i - 1]) / (sim.h_t1grid[i - 1] - sim.h_t1grid[i - 2]);
    std::cout << dmfe::console::INFO() << "Successfully loaded binary simulation data from " << filename << "\nTime: " << sim.h_t1grid.back() << ", Loop: " << loaded_params.loop << ", Energy: " << file_energy << std::endl;
    return true;
}

#if defined(H5_RUNTIME_OPTIONAL) || defined(USE_HDF5)
bool loadSimulationStateHDF5(const std::string &filename, SimulationData &sim,
                            int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param,
                            size_t len_param, double delta_t_min_param, double delta_max_param,
                            bool use_serk2_param,
                            LoadedStateParams& loaded_params)
{
#if defined(H5_RUNTIME_OPTIONAL)
    if (!h5rt::available()) return false;
    auto file = h5rt::open_file_readonly(filename.c_str());
    if (file < 0) return false;
    double file_T0=0, file_lambda=0, file_Gamma=0; int file_p=0, file_p2=0, file_len=0;
    double file_delta_t_min=0, file_delta_t_max=0; int file_use_serk2=0; int file_log_response_interp=0;
    // Default tail_fit_enabled to current config for backward compatibility when attribute is absent
    int file_tail_fit_enabled = (config.tail_fit_enabled ? 1 : 0);
    if (!h5rt::read_attr_double(file, "T0", file_T0) ||
        !h5rt::read_attr_double(file, "lambda", file_lambda) ||
        !h5rt::read_attr_int(file, "p", file_p) ||
        !h5rt::read_attr_int(file, "p2", file_p2)) { h5rt::close_file(file); return false; }
    // Read additional attributes if present
    h5rt::read_attr_double(file, "Gamma", file_Gamma);
    h5rt::read_attr_int(file, "len", file_len);
    h5rt::read_attr_double(file, "delta_t_min", file_delta_t_min);
    h5rt::read_attr_double(file, "delta_max", file_delta_t_max);
    h5rt::read_attr_int(file, "use_serk2", file_use_serk2);
    // Optional attribute present in newer files
    h5rt::read_attr_int(file, "log_response_interp", file_log_response_interp);
    // Optional attribute: tail fit flag
    h5rt::read_attr_int(file, "tail_fit_enabled", file_tail_fit_enabled);
    if (file_p != p_param || file_p2 != p2_param || std::abs(file_lambda - lambda_param) > 1e-10 || std::abs(file_T0 - T0_param) > 1e-10 ||
        std::abs(file_Gamma - Gamma_param) > 1e-10 || file_len != (int)len_param || 
    std::abs(file_delta_t_min - delta_t_min_param) > 1e-10 || std::abs(file_delta_t_max - delta_max_param) > 1e-10 ||
    file_use_serk2 != (use_serk2_param ? 1 : 0) ||
    ((file_log_response_interp != 0) != (config.log_response_interp ? true : false)) ||
    ((file_tail_fit_enabled != 0) != (config.tail_fit_enabled ? true : false))) {
    std::cerr << dmfe::console::WARN() << "File parameters don't match current simulation parameters" << std::endl;
        h5rt::close_file(file);
        return false;
    }
    double file_delta=0, file_delta_t=0; int file_loop=0; double file_energy=0.0;
    (void)h5rt::read_attr_double(file, "delta", file_delta);
    (void)h5rt::read_attr_double(file, "delta_t", file_delta_t);
    (void)h5rt::read_attr_int(file, "iteration", file_loop);
    (void)h5rt::read_attr_double(file, "energy", file_energy);

    size_t t1grid_size = h5rt::dataset_length(file, "t1grid");
    size_t qkv_size = h5rt::dataset_length(file, "QKv");
    if (t1grid_size == 0 || qkv_size == 0 || (qkv_size % t1grid_size) != 0) {
        std::cerr << dmfe::console::ERR() << "Inconsistent dimensions in HDF5 file" << std::endl;
        h5rt::close_file(file); return false;
    }
    size_t vector_len = qkv_size / t1grid_size;
    sim.h_t1grid.resize(t1grid_size);
    sim.h_QKv.resize(t1grid_size * vector_len);
    sim.h_QRv.resize(t1grid_size * vector_len);
    sim.h_dQKv.resize(t1grid_size * vector_len);
    sim.h_dQRv.resize(t1grid_size * vector_len);
    sim.h_rvec.resize(t1grid_size);
    sim.h_drvec.resize(t1grid_size);
    if (!h5rt::read_dataset_1d_double(file, "t1grid", sim.h_t1grid) ||
        !h5rt::read_dataset_1d_double(file, "QKv", sim.h_QKv) ||
        !h5rt::read_dataset_1d_double(file, "QRv", sim.h_QRv) ||
        !h5rt::read_dataset_1d_double(file, "dQKv", sim.h_dQKv) ||
        !h5rt::read_dataset_1d_double(file, "dQRv", sim.h_dQRv) ||
        !h5rt::read_dataset_1d_double(file, "rvec", sim.h_rvec)) {
        h5rt::close_file(file); return false;
    }
    (void)h5rt::read_dataset_1d_double(file, "drvec", sim.h_drvec);
    loaded_params.delta = file_delta; loaded_params.delta_t = file_delta_t; loaded_params.loop = file_loop;
    sim.h_delta_t_ratio.resize(t1grid_size);
    sim.h_delta_t_ratio[0] = 0.0;
    for (size_t i = 2; i < t1grid_size; ++i)
        sim.h_delta_t_ratio[i] = (sim.h_t1grid[i] - sim.h_t1grid[i - 1]) / (sim.h_t1grid[i - 1] - sim.h_t1grid[i - 2]);
    std::cout << "Successfully loaded HDF5 simulation data from " << filename << "\nTime: " << sim.h_t1grid.back() << ", Loop: " << loaded_params.loop;
    if (file_energy != 0.0) std::cout << ", Energy: " << file_energy;
    std::cout << std::endl;
    h5rt::close_file(file);
    return true;
#else
    // Original C++ API path
    try
    {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        double file_T0, file_lambda, file_Gamma = 0; int file_p, file_p2, file_len = 0;
    double file_delta_t_min = 0, file_delta_t_max = 0; int file_use_serk2 = 0; int file_log_response_interp = 0;
        // Default tail_fit_enabled to current config for backward compatibility when attribute is absent
        int file_tail_fit_enabled = (config.tail_fit_enabled ? 1 : 0);
        file.openAttribute("T0").read(H5::PredType::NATIVE_DOUBLE, &file_T0);
        file.openAttribute("lambda").read(H5::PredType::NATIVE_DOUBLE, &file_lambda);
        file.openAttribute("p").read(H5::PredType::NATIVE_INT, &file_p);
        file.openAttribute("p2").read(H5::PredType::NATIVE_INT, &file_p2);
        // Read additional attributes if present
        try { file.openAttribute("Gamma").read(H5::PredType::NATIVE_DOUBLE, &file_Gamma); } catch (...) {}
        try { file.openAttribute("len").read(H5::PredType::NATIVE_INT, &file_len); } catch (...) {}
        try { file.openAttribute("delta_t_min").read(H5::PredType::NATIVE_DOUBLE, &file_delta_t_min); } catch (...) {}
        try { file.openAttribute("delta_max").read(H5::PredType::NATIVE_DOUBLE, &file_delta_t_max); } catch (...) {}
    try { file.openAttribute("use_serk2").read(H5::PredType::NATIVE_INT, &file_use_serk2); } catch (...) {}
    try { file.openAttribute("log_response_interp").read(H5::PredType::NATIVE_INT, &file_log_response_interp); } catch (...) {}
        // Optional attribute: tail fit flag
        try { file.openAttribute("tail_fit_enabled").read(H5::PredType::NATIVE_INT, &file_tail_fit_enabled); } catch (...) {}
    // aggressive_sparsify attribute deprecated; ignore if present
        if (file_p != p_param || file_p2 != p2_param || fabs(file_lambda - lambda_param) > 1e-10 || fabs(file_T0 - T0_param) > 1e-10 ||
            fabs(file_Gamma - Gamma_param) > 1e-10 || file_len != (int)len_param || 
            fabs(file_delta_t_min - delta_t_min_param) > 1e-10 || fabs(file_delta_t_max - delta_max_param) > 1e-10 ||
            file_use_serk2 != (use_serk2_param ? 1 : 0) ||
            ((file_log_response_interp != 0) != (config.log_response_interp ? true : false)) ||
            ((file_tail_fit_enabled != 0) != (config.tail_fit_enabled ? true : false))) {
        std::cerr << dmfe::console::WARN() << "File parameters don't match current simulation parameters" << std::endl;
            return false;
        }
        double file_delta=0, file_delta_t=0; int file_loop=0; double file_energy=0.0;
        file.openAttribute("delta").read(H5::PredType::NATIVE_DOUBLE, &file_delta);
        file.openAttribute("delta_t").read(H5::PredType::NATIVE_DOUBLE, &file_delta_t);
        file.openAttribute("iteration").read(H5::PredType::NATIVE_INT, &file_loop);
        try { file.openAttribute("energy").read(H5::PredType::NATIVE_DOUBLE, &file_energy); } catch (...) {}
        H5::DataSet t1_dataset = file.openDataSet("t1grid");
        H5::DataSpace t1_space = t1_dataset.getSpace();
        hsize_t t1_dims[1]; t1_space.getSimpleExtentDims(t1_dims, nullptr);
        size_t t1grid_size = t1_dims[0];
        H5::DataSet qkv_dataset = file.openDataSet("QKv");
        H5::DataSpace qkv_space = qkv_dataset.getSpace();
        hsize_t qkv_dims[1]; qkv_space.getSimpleExtentDims(qkv_dims, nullptr);
    if (qkv_dims[0] % t1grid_size != 0) { std::cerr << dmfe::console::ERR() << "Inconsistent dimensions in HDF5 file" << std::endl; return false; }
        size_t vector_len = qkv_dims[0] / t1grid_size;
        sim.h_t1grid.resize(t1grid_size);
        sim.h_QKv.resize(t1grid_size * vector_len);
        sim.h_QRv.resize(t1grid_size * vector_len);
        sim.h_dQKv.resize(t1grid_size * vector_len);
        sim.h_dQRv.resize(t1grid_size * vector_len);
        sim.h_rvec.resize(t1grid_size);
        sim.h_drvec.resize(t1grid_size);
        t1_dataset.read(sim.h_t1grid.data(), H5::PredType::NATIVE_DOUBLE);
        qkv_dataset.read(sim.h_QKv.data(), H5::PredType::NATIVE_DOUBLE);
        file.openDataSet("QRv").read(sim.h_QRv.data(), H5::PredType::NATIVE_DOUBLE);
        file.openDataSet("dQKv").read(sim.h_dQKv.data(), H5::PredType::NATIVE_DOUBLE);
        file.openDataSet("dQRv").read(sim.h_dQRv.data(), H5::PredType::NATIVE_DOUBLE);
        file.openDataSet("rvec").read(sim.h_rvec.data(), H5::PredType::NATIVE_DOUBLE);
        try { file.openDataSet("drvec").read(sim.h_drvec.data(), H5::PredType::NATIVE_DOUBLE); } catch (...) {}
        loaded_params.delta = file_delta; loaded_params.delta_t = file_delta_t; loaded_params.loop = file_loop;
        sim.h_delta_t_ratio.resize(t1grid_size);
        sim.h_delta_t_ratio[0] = 0.0;
        for (size_t i = 2; i < t1grid_size; ++i)
            sim.h_delta_t_ratio[i] = (sim.h_t1grid[i] - sim.h_t1grid[i - 1]) / (sim.h_t1grid[i - 1] - sim.h_t1grid[i - 2]);
    std::cout << dmfe::console::INFO() << "Successfully loaded HDF5 simulation data from " << filename << "\nTime: " << sim.h_t1grid.back() << ", Loop: " << loaded_params.loop;
    if (file_energy != 0.0) std::cout << ", Energy: " << file_energy;
    std::cout << std::endl;
        return true;
    }
    catch (H5::Exception &e) {
    std::cerr << dmfe::console::ERR() << "HDF5 error: " << e.getCDetailMsg() << std::endl;
        return false;
    }
#endif
}

#endif // defined(H5_RUNTIME_OPTIONAL) || defined(USE_HDF5)

bool checkParametersMatch(const std::string &paramFilename, int p_param, int p2_param, double lambda_param, 
                         double T0_param, double Gamma_param, size_t len_param, double delta_t_min_param, double delta_max_param,
                         bool use_serk2_param)
{
    // Helper: parse simple key = value file into map
    auto parse_kv_file = [](const std::string& path, std::unordered_map<std::string, std::string>& out) -> bool {
        out.clear();
        std::ifstream in(path);
        if (!in) return false;
        std::string line;
        auto trim = [](const std::string& s){ auto a = s.find_first_not_of(" \t\r\n"); if (a==std::string::npos) return std::string(); auto b = s.find_last_not_of(" \t\r\n"); return s.substr(a, b-a+1); };
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string k = trim(line.substr(0, eq));
            std::string v = trim(line.substr(eq + 1));
            if (!k.empty()) out[k] = v;
        }
        return !out.empty();
    };
    auto format_double = [](double d) -> std::string {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(12) << d;
        return oss.str();
    };
    std::ifstream paramFile(paramFilename);
    if (!paramFile)
    {
    std::cerr << dmfe::console::ERR() << "Could not open parameter file " << paramFilename << std::endl;
        return false;
    }
    int file_p = -1, file_p2 = -1, file_len = -1;
    double file_lambda = -1.0, file_T0 = -1.0, file_Gamma = -1.0, file_delta_t_min = -1.0, file_delta_max = -1.0;
    bool file_use_serk2 = false;
    bool file_log_response_interp = false; // default for legacy files
    bool have_file_log_response_interp = false;
    bool file_tail_fit_enabled = true; // default (current code default true)
    bool have_file_tail_fit_enabled = false; // track presence
    int file_sparsify_sweeps = -1; // default: -1 (auto)
    bool have_file_sparsify_sweeps = false; // track presence to preserve backward-compat
    // Capture saved grid provenance mirrored into params.txt (if present)
    std::unordered_map<std::string, std::string> saved_grid; // keys without prefix: len, Tmax, spline_order, interp_method, interp_order, fh_stencil, alpha, delta
    std::string line;
    while (std::getline(paramFile, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream iss(line);
        std::string name, eq;
        iss >> name >> eq;
        if (eq != "=")
            continue;
        if (name == "p")
            iss >> file_p;
        else if (name == "p2")
            iss >> file_p2;
        else if (name == "lambda")
            iss >> file_lambda;
        else if (name == "T0")
        {
            std::string t;
            iss >> t;
            file_T0 = (t == "inf" ? 1e50 : std::stod(t));
        }
        else if (name == "Gamma")
            iss >> file_Gamma;
        else if (name == "len")
            iss >> file_len;
        else if (name == "delta_t_min")
            iss >> file_delta_t_min;
        else if (name == "delta_max")
            iss >> file_delta_max;
        else if (name == "use_serk2") {
            std::string val;
            iss >> val;
            file_use_serk2 = (val == "true");
        }
        else if (name == "sparsify_sweeps") {
            iss >> file_sparsify_sweeps;
            have_file_sparsify_sweeps = true;
        }
        else if (name == "log_response_interp") {
            std::string val; iss >> val;
            have_file_log_response_interp = true;
            file_log_response_interp = (val == "true");
        }
        else if (name == "tail_fit_enabled") {
            std::string val; iss >> val;
            have_file_tail_fit_enabled = true;
            file_tail_fit_enabled = (val == "true");
        }
        // Capture any grid_* entries mirrored from grid_params.txt into saved_grid
        else if (name.rfind("grid_", 0) == 0) {
            std::string val;
            std::getline(iss, val);
            // consume leading spaces if any
            auto trim = [](const std::string& s){ auto a = s.find_first_not_of(" \t\r\n"); if (a==std::string::npos) return std::string(); auto b = s.find_last_not_of(" \t\r\n"); return s.substr(a, b-a+1); };
            val = trim(val);
            if (name == "grid_len") {
                saved_grid["len"] = val;
            } else if (name == "grid_Tmax") {
                saved_grid["Tmax"] = val;
            } else if (name == "grid_spline_order") {
                saved_grid["spline_order"] = val;
            } else if (name == "grid_interp_method") {
                saved_grid["interp_method"] = val;
            } else if (name == "grid_interp_order") {
                saved_grid["interp_order"] = val;
            } else if (name == "grid_fh_stencil") {
                saved_grid["fh_stencil"] = val;
            } else if (name == "grid_alpha") {
                saved_grid["alpha"] = val;
            } else if (name == "grid_delta") {
                saved_grid["delta"] = val;
            }
        }
    }
    bool match = true;
    std::string dir = paramFilename.substr(0, paramFilename.find_last_of('/'));
    auto mismatch = [&](const std::string &n, const std::string &a, const std::string &b)
    {
        if (config.debug) {
            std::cerr << dmfe::console::WARN() << "Parameter mismatch in " << dir << ": " << n
                      << " (file: " << a << ", current: " << b << ")" << std::endl;
        }
        match = false;
    };
    if (file_p != p_param)
        mismatch("p", std::to_string(file_p), std::to_string(p_param));
    if (file_p2 != p2_param)
        mismatch("p2", std::to_string(file_p2), std::to_string(p2_param));
    if (std::abs(file_lambda - lambda_param) > 1e-10)
        mismatch("lambda", format_double(file_lambda), format_double(lambda_param));
    if (std::abs(file_T0 - T0_param) > 1e-10)
        mismatch("T0", (file_T0 >= 1e50 ? "inf" : format_double(file_T0)), (T0_param >= 1e50 ? "inf" : format_double(T0_param)));
    if (std::abs(file_Gamma - Gamma_param) > 1e-10)
        mismatch("Gamma", format_double(file_Gamma), format_double(Gamma_param));
    if (file_len != (int)len_param)
        mismatch("len", std::to_string(file_len), std::to_string(len_param));
    if (std::abs(file_delta_t_min - delta_t_min_param) > 1e-10)
        mismatch("delta_t_min", format_double(file_delta_t_min), format_double(delta_t_min_param));
    if (file_delta_max > 0 && std::abs(file_delta_max - delta_max_param) / std::max(file_delta_max, delta_max_param) > 1e-10)
        mismatch("delta_max", format_double(file_delta_max), format_double(delta_max_param));
    if (file_use_serk2 != use_serk2_param)
        mismatch("use_serk2", file_use_serk2 ? "true" : "false", use_serk2_param ? "true" : "false");
    // Compare log_response_interp when present in params.txt; skip if absent for backward compatibility
    if (have_file_log_response_interp && file_log_response_interp != config.log_response_interp) {
        mismatch("log_response_interp", file_log_response_interp ? "true" : "false", config.log_response_interp ? "true" : "false");
    }
    // Compare sparsify_sweeps when present in params.txt; skip if absent to keep legacy files loadable
    if (have_file_sparsify_sweeps && file_sparsify_sweeps != config.sparsify_sweeps) {
        mismatch("sparsify_sweeps", std::to_string(file_sparsify_sweeps), std::to_string(config.sparsify_sweeps));
    }
    // Compare tail_fit_enabled when present (backward-compatible skip if absent)
    if (have_file_tail_fit_enabled && file_tail_fit_enabled != config.tail_fit_enabled) {
        mismatch("tail_fit_enabled", file_tail_fit_enabled ? "true" : "false", config.tail_fit_enabled ? "true" : "false");
    }

    // Additional: check grid parameters used for this saved run vs the currently available grid
    // Strategy:
    // 1) Prefer comparing saved params.txt grid_* values against the current Grid_data/<len>/grid_params.txt.
    // 2) If Grid_data file is missing, fallback to comparing saved values against runtime config (len, tmax) only.
    {
        // Load current grid (by len) if available
        std::ostringstream gpPath;
        gpPath << "Grid_data/" << len_param << "/grid_params.txt";
        std::unordered_map<std::string, std::string> gp; // current grid provenance
        bool have_gp = parse_kv_file(gpPath.str(), gp);

        auto get_gp = [&](const char* k) -> std::string {
            auto it = gp.find(k); return (it != gp.end() ? it->second : std::string()); };

        // Compare len
        if (saved_grid.count("len")) {
            // saved value must match the current grid (if present) otherwise the runtime len
            long long saved_len = -1; try { saved_len = std::stoll(saved_grid["len"]); } catch (...) {}
            if (have_gp && !get_gp("len").empty()) {
                long long gp_len = -1; try { gp_len = std::stoll(get_gp("len")); } catch (...) {}
                if (saved_len != gp_len) mismatch("grid_len", std::to_string(saved_len), std::to_string(gp_len));
            } else {
                if (saved_len != (long long)len_param) mismatch("grid_len", std::to_string(saved_len), std::to_string(len_param));
            }
        }

        // Compare Tmax (tolerant relative)
        if (saved_grid.count("Tmax")) {
            double saved_T = std::numeric_limits<double>::quiet_NaN();
            try { saved_T = std::stod(saved_grid["Tmax"]); } catch (...) {}
            if (have_gp && !get_gp("Tmax").empty()) {
                double gpT = std::numeric_limits<double>::quiet_NaN();
                try { gpT = std::stod(get_gp("Tmax")); } catch (...) {}
                if (std::isfinite(saved_T) && std::isfinite(gpT)) {
                    double denom = std::max(std::abs(saved_T), std::abs(gpT));
                    if (denom == 0) denom = 1.0;
                    if (std::abs(saved_T - gpT) / denom > 1e-10)
                        mismatch("grid_Tmax", format_double(saved_T), format_double(gpT));
                }
            } else {
                double curT = config.tmax;
                if (std::isfinite(saved_T) && std::isfinite(curT)) {
                    double denom = std::max(std::abs(saved_T), std::abs(curT));
                    if (denom == 0) denom = 1.0;
                    if (std::abs(saved_T - curT) / denom > 1e-10)
                        mismatch("grid_Tmax", format_double(saved_T), format_double(curT));
                }
            }
        }

        // Compare optional fh_stencil if present on both sides (exact string)
        if (saved_grid.count("fh_stencil")) {
            if (have_gp && !get_gp("fh_stencil").empty()) {
                auto a = saved_grid["fh_stencil"]; auto b = get_gp("fh_stencil");
                if (a != b) mismatch("grid_fh_stencil", a, b);
            }
        }

        // Compare alpha (tolerant relative/abs): allow tiny differences
        if (saved_grid.count("alpha")) {
            double sa = std::numeric_limits<double>::quiet_NaN();
            try { sa = std::stod(saved_grid["alpha"]); } catch (...) {}
            if (have_gp && !get_gp("alpha").empty()) {
                double ga = std::numeric_limits<double>::quiet_NaN();
                try { ga = std::stod(get_gp("alpha")); } catch (...) {}
                if (std::isfinite(sa) && std::isfinite(ga)) {
                    double diff = std::abs(sa - ga);
                    double denom = std::max(std::abs(sa), std::abs(ga));
                    if (denom == 0) denom = 1.0;
                    if (diff > 1e-12 && diff/denom > 1e-10) mismatch("grid_alpha", format_double(sa), format_double(ga));
                }
            }
        }

        // Compare delta (tolerant relative/abs)
        if (saved_grid.count("delta")) {
            double sd = std::numeric_limits<double>::quiet_NaN();
            try { sd = std::stod(saved_grid["delta"]); } catch (...) {}
            if (have_gp && !get_gp("delta").empty()) {
                double gd = std::numeric_limits<double>::quiet_NaN();
                try { gd = std::stod(get_gp("delta")); } catch (...) {}
                if (std::isfinite(sd) && std::isfinite(gd)) {
                    double diff = std::abs(sd - gd);
                    double denom = std::max(std::abs(sd), std::abs(gd));
                    if (denom == 0) denom = 1.0;
                    if (diff > 1e-12 && diff/denom > 1e-10) mismatch("grid_delta", format_double(sd), format_double(gd));
                }
            }
        }

        // Compare spline/interp only when both sides present; strings are compared case-sensitively
        if (saved_grid.count("spline_order")) {
            if (have_gp && !get_gp("spline_order").empty()) {
                auto a = saved_grid["spline_order"]; auto b = get_gp("spline_order");
                if (a != b) mismatch("grid_spline_order", a, b);
            }
        }
        if (saved_grid.count("interp_method")) {
            if (have_gp && !get_gp("interp_method").empty()) {
                auto a = saved_grid["interp_method"]; auto b = get_gp("interp_method");
                if (a != b) mismatch("grid_interp_method", a, b);
            }
        }
        if (saved_grid.count("interp_order")) {
            if (have_gp && !get_gp("interp_order").empty()) {
                auto a = saved_grid["interp_order"]; auto b = get_gp("interp_order");
                if (a != b) mismatch("grid_interp_order", a, b);
            }
        }

        if (!have_gp && config.debug && !saved_grid.empty()) {
            std::cerr << dmfe::console::WARN() << "Grid parameters file missing: " << gpPath.str() << ". Compared against runtime config where possible." << std::endl;
        }
    }
    return match;
}

bool loadSimulationState(const std::string &filename, SimulationData &sim,
                        int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param, 
                        size_t len_param, double delta_t_min_param, double delta_max_param,
                        bool use_serk2_param,
                        LoadedStateParams& loaded_params)
{
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    std::string paramFilename = dirPath + "/params.txt";
    if (fileExists(paramFilename)) {
        if (!checkVersionCompatibilityInteractive(paramFilename)) {
            std::cerr << dmfe::console::ERR() << "Loading cancelled due to version mismatch." << std::endl;
            return false;
        }
    if (!checkParametersMatch(paramFilename, p_param, p2_param, lambda_param, T0_param, Gamma_param, len_param, delta_t_min_param, delta_max_param, use_serk2_param)) {
            std::cerr << dmfe::console::ERR() << "Parameter mismatch: Will not load file " << filename << std::endl;
            return false;
        }
    std::cout << dmfe::console::INFO() << "All parameters match. Proceeding with loading..." << std::endl;
    } else {
    std::cerr << dmfe::console::WARN() << "Parameter file " << paramFilename << " not found. Will attempt to load but compatibility cannot be verified." << std::endl;
    }

#if defined(H5_RUNTIME_OPTIONAL)
    if (fileExists(filename) && h5rt::available()) {
        if (loadSimulationStateHDF5(filename, sim, p_param, p2_param, lambda_param, T0_param, Gamma_param, len_param, delta_t_min_param, delta_max_param, use_serk2_param, loaded_params)) {
            return true;
        }
    std::cerr << dmfe::console::WARN() << "HDF5 loading failed; trying binary format..." << std::endl;
    }
#elif defined(USE_HDF5)
    if (fileExists(filename)) {
        try {
            if (loadSimulationStateHDF5(filename, sim, p_param, p2_param, lambda_param, T0_param, Gamma_param, len_param, delta_t_min_param, delta_max_param, use_serk2_param, loaded_params)) {
                return true;
            }
        } catch (H5::Exception &e) {
            std::cerr << dmfe::console::WARN() << "HDF5 loading failed: " << e.getCDetailMsg() << "\nTrying binary format..." << std::endl;
        }
    }
#endif

    std::string binFilename = dirPath + "/data.bin";
    if (fileExists(binFilename)) {
        if (loadSimulationStateBinary(binFilename, sim, p_param, p2_param, lambda_param, T0_param, Gamma_param, len_param, delta_t_min_param, delta_max_param, use_serk2_param, loaded_params)) {
            return true;
        }
    }
    return false;
}
