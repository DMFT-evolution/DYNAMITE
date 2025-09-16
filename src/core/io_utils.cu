#include "io_utils.hpp"
#include "simulation_data.hpp"
#include "gpu_memory_utils.hpp"
#include "math_ops.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "convolution.hpp"
#include "device_utils.cuh"
#include "time_steps.hpp"
#include "version_info.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <cmath>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <unistd.h>
#include <limits.h>
#include <errno.h>
#include <cstring>
#include <chrono>
#include <vector>

#if defined(H5_RUNTIME_OPTIONAL)
#include "io/h5_runtime.hpp"
#elif defined(USE_HDF5)
#include "H5Cpp.h"
#endif

#include "version_compat.hpp"

using namespace std;

// External global variables
extern SimulationConfig config;
extern SimulationData* sim;
extern RKData* rk;
extern size_t peak_memory_kb;
extern size_t peak_gpu_memory_mb;
extern std::chrono::high_resolution_clock::time_point program_start_time;

// All former global vectors now stored inside SimulationData passed to functions.

// Basic imports
std::vector<double> importVectorFromFile(const std::string &filename)
{
    std::vector<double> data;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
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
        std::cerr << "Error: Could not open file " << filename << std::endl;
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

// Paths
std::string getParameterDirPath(const std::string& resultsDir_param, int p_param, int p2_param, 
                               double lambda_param, double T0_param, double Gamma_param, size_t len_param)
{
    std::ostringstream d;
    d << std::fixed << std::setprecision(2) << resultsDir_param
      << "p=" << p_param << "_p2=" << p2_param << "_lambda=" << lambda_param
      << "_T0=" << (T0_param >= 1e10 ? "inf" : std::to_string(T0_param))
      << "_G=" << Gamma_param << "_len=" << len_param;
    return d.str();
}

void ensureDirectoryExists(const std::string &dir)
{
    struct stat st{};
    if (stat(dir.c_str(), &st) == -1)
    {
        std::string cmd = "mkdir -p '" + dir + "'";
        if (system(cmd.c_str()) != 0)
            std::cerr << "Warning: Could not create directory " << dir << std::endl;
    }
}

std::string getFilename(const std::string& resultsDir_param, int p_param, int p2_param, 
                       double lambda_param, double T0_param, double Gamma_param, size_t len_param, bool save_output_param)
{
    auto path = getParameterDirPath(resultsDir_param, p_param, p2_param, lambda_param, T0_param, Gamma_param, len_param);
    if (save_output_param)
        ensureDirectoryExists(path);
    return path + "/data.h5";
}

bool fileExists(const std::string &filename)
{
    std::ifstream f(filename);
    return f.good();
}

// Binary load
bool loadSimulationStateBinary(const std::string &filename, SimulationData &sim,
                              int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param,
                              LoadedStateParams& loaded_params)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Error: Could not open binary file " << filename << std::endl;
        return false;
    }
    int header_version;
    file.read(reinterpret_cast<char *>(&header_version), sizeof(int));
    if (header_version != 1)
    {
        std::cerr << "Error: Unsupported binary file version: " << header_version << std::endl;
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
        std::cerr << "Warning: File parameters don't match current simulation parameters" << std::endl;
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
    std::cout << "Successfully loaded binary simulation data from " << filename << "\nTime: " << sim.h_t1grid.back() << ", Loop: " << loaded_params.loop << ", Energy: " << file_energy << std::endl;
    return true;
}

#if defined(H5_RUNTIME_OPTIONAL) || defined(USE_HDF5)
// Forward-declare save to satisfy references before its definition below
void saveSimulationStateHDF5(const std::string& filename, double delta, double delta_t);

bool loadSimulationStateHDF5(const std::string &filename, SimulationData &sim,
                            int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param,
                            LoadedStateParams& loaded_params)
{
#if defined(H5_RUNTIME_OPTIONAL)
    if (!h5rt::available()) return false;
    auto file = h5rt::open_file_readonly(filename.c_str());
    if (file < 0) return false;
    double file_T0=0, file_lambda=0; int file_p=0, file_p2=0;
    if (!h5rt::read_attr_double(file, "T0", file_T0) ||
        !h5rt::read_attr_double(file, "lambda", file_lambda) ||
        !h5rt::read_attr_int(file, "p", file_p) ||
        !h5rt::read_attr_int(file, "p2", file_p2)) { h5rt::close_file(file); return false; }
    if (file_p != p_param || file_p2 != p2_param || std::abs(file_lambda - lambda_param) > 1e-10 || std::abs(file_T0 - T0_param) > 1e-10) {
        std::cerr << "Warning: File parameters don't match current simulation parameters" << std::endl;
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
        std::cerr << "Error: Inconsistent dimensions in HDF5 file" << std::endl;
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
        double file_T0, file_lambda; int file_p, file_p2;
        file.openAttribute("T0").read(H5::PredType::NATIVE_DOUBLE, &file_T0);
        file.openAttribute("lambda").read(H5::PredType::NATIVE_DOUBLE, &file_lambda);
        file.openAttribute("p").read(H5::PredType::NATIVE_INT, &file_p);
        file.openAttribute("p2").read(H5::PredType::NATIVE_INT, &file_p2);
        if (file_p != p_param || file_p2 != p2_param || fabs(file_lambda - lambda_param) > 1e-10 || fabs(file_T0 - T0_param) > 1e-10) {
            std::cerr << "Warning: File parameters don't match current simulation parameters" << std::endl;
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
        if (qkv_dims[0] % t1grid_size != 0) { std::cerr << "Error: Inconsistent dimensions in HDF5 file" << std::endl; return false; }
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
        std::cout << "Successfully loaded HDF5 simulation data from " << filename << "\nTime: " << sim.h_t1grid.back() << ", Loop: " << loaded_params.loop;
        if (file_energy != 0.0) std::cout << ", Energy: " << file_energy;
        std::cout << std::endl;
        return true;
    }
    catch (H5::Exception &e) {
        std::cerr << "HDF5 error: " << e.getCDetailMsg() << std::endl;
        return false;
    }
#endif
}

#endif // defined(H5_RUNTIME_OPTIONAL) || defined(USE_HDF5)

bool checkParametersMatch(const std::string &paramFilename, int p_param, int p2_param, double lambda_param, 
                         double T0_param, double Gamma_param, size_t len_param, double delta_t_min_param, double delta_max_param)
{
    std::ifstream paramFile(paramFilename);
    if (!paramFile)
    {
        std::cerr << "Error: Could not open parameter file " << paramFilename << std::endl;
        return false;
    }
    int file_p = -1, file_p2 = -1, file_len = -1;
    double file_lambda = -1.0, file_T0 = -1.0, file_Gamma = -1.0, file_delta_t_min = -1.0, file_delta_max = -1.0;
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
    }
    bool match = true;
    auto mismatch = [&](const std::string &n, const std::string &a, const std::string &b)
    { std::cerr<<"Parameter mismatch: "<<n<<" (file: "<<a<<", current: "<<b<<")"<<std::endl; match=false; };
    if (file_p != p_param)
        mismatch("p", std::to_string(file_p), std::to_string(p_param));
    if (file_p2 != p2_param)
        mismatch("p2", std::to_string(file_p2), std::to_string(p2_param));
    if (std::abs(file_lambda - lambda_param) > 1e-10)
        mismatch("lambda", std::to_string(file_lambda), std::to_string(lambda_param));
    if (std::abs(file_T0 - T0_param) > 1e-10)
        mismatch("T0", (file_T0 >= 1e50 ? "inf" : std::to_string(file_T0)), (T0_param >= 1e50 ? "inf" : std::to_string(T0_param)));
    if (std::abs(file_Gamma - Gamma_param) > 1e-10)
        mismatch("Gamma", std::to_string(file_Gamma), std::to_string(Gamma_param));
    if (file_len != (int)len_param)
        mismatch("len", std::to_string(file_len), std::to_string(len_param));
    if (std::abs(file_delta_t_min - delta_t_min_param) > 1e-10)
        mismatch("delta_t_min", std::to_string(file_delta_t_min), std::to_string(delta_t_min_param));
    if (file_delta_max > 0 && std::abs(file_delta_max - delta_max_param) / std::max(file_delta_max, delta_max_param) > 1e-10)
        mismatch("delta_max", std::to_string(file_delta_max), std::to_string(delta_max_param));
    return match;
}

bool loadSimulationState(const std::string &filename, SimulationData &sim,
                        int p_param, int p2_param, double lambda_param, double T0_param, double Gamma_param, 
                        size_t len_param, double delta_t_min_param, double delta_max_param,
                        LoadedStateParams& loaded_params)
{
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    std::string paramFilename = dirPath + "/params.txt";
    if (fileExists(paramFilename)) {
        if (!checkVersionCompatibilityInteractive(paramFilename)) {
            std::cerr << "Loading cancelled due to version mismatch." << std::endl;
            return false;
        }
        if (!checkParametersMatch(paramFilename, p_param, p2_param, lambda_param, T0_param, Gamma_param, len_param, delta_t_min_param, delta_max_param)) {
            std::cerr << "Parameter mismatch: Will not load file " << filename << std::endl;
            return false;
        }
        std::cout << "All parameters match. Proceeding with loading..." << std::endl;
    } else {
        std::cerr << "Warning: Parameter file " << paramFilename << " not found. Will attempt to load but compatibility cannot be verified." << std::endl;
    }

#if defined(H5_RUNTIME_OPTIONAL)
    if (fileExists(filename) && h5rt::available()) {
        if (loadSimulationStateHDF5(filename, sim, p_param, p2_param, lambda_param, T0_param, Gamma_param, loaded_params))
            return true;
        std::cerr << "HDF5 loading failed; trying binary format..." << std::endl;
    }
#elif defined(USE_HDF5)
    if (fileExists(filename)) {
        try {
            return loadSimulationStateHDF5(filename, sim, p_param, p2_param, lambda_param, T0_param, Gamma_param, loaded_params);
        } catch (H5::Exception &e) {
            std::cerr << "HDF5 loading failed: " << e.getCDetailMsg() << "\nTrying binary format..." << std::endl;
        }
    }
#endif

    std::string binFilename = dirPath + "/data.bin";
    if (fileExists(binFilename))
        return loadSimulationStateBinary(binFilename, sim, p_param, p2_param, lambda_param, T0_param, Gamma_param, loaded_params);
    return false;
}

// Helper functions for saveHistory
void SigmaK(const std::vector<double>& qk, std::vector<double>& result)
{
    for (size_t i = 0; i < qk.size(); ++i) {
        result[i] = Dflambda(qk[i]);
    }
}

// GPU kernel for computing energy at all time points efficiently
__global__ void computeEnergyHistoryKernel(
    const double* __restrict__ QKv,
    const double* __restrict__ QRv,
    const double* __restrict__ integ,
    const double* __restrict__ theta,
    const double* __restrict__ t1grid,
    double* __restrict__ energy_history,
    double T0,
    size_t len,
    size_t t1len)
{
    extern __shared__ double sdata[];
    double* integ_shared = sdata;
    double* reduction_shared = &sdata[len];
    
    int t_idx = blockIdx.x;  // Each block handles one time point
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    
    if (t_idx >= t1len) return;
    
    // Load integ into shared memory once per block
    for (int i = tid; i < len; i += nthreads) {
        integ_shared[i] = integ[i];
    }
    __syncthreads();
    
    // Compute sigmaK and convolution in one pass
    double sum = 0.0;
    double t = t1grid[t_idx];
    size_t base_idx = t_idx * len;
    
    for (int i = tid; i < len; i += nthreads) {
        double qk = QKv[base_idx + i];
        double qr = QRv[base_idx + i];
        double sigmaK = DflambdaGPU(qk);
        
        // Match the ConvAGPU scaling approach
        sum += sigmaK * qr * integ_shared[i];
        
        // Add direct term for i=0
        if (i == 0) {
            sum += sigmaK / T0;
        }
    }
    
    // Block-level reduction
    reduction_shared[tid] = sum;
    __syncthreads();
    
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction_shared[tid] += reduction_shared[tid + stride];
        }
        __syncthreads();
    }
    
    // Write final energy (negative because it's -energy)
    if (tid == 0) {
        energy_history[t_idx] = -t * reduction_shared[0];
    }
}

void saveHistory(const std::string& filename, double delta, double delta_t, 
                 SimulationData& simulation, size_t len_param, double T0_param, bool gpu_param) {
    // Ensure data is on CPU for saving
    if (gpu_param) {
        copyVectorsToCPU(simulation);
    }
    
    size_t t1len = simulation.h_t1grid.size();
    std::vector<double> energy_history(t1len);
    std::vector<double> qk0_history(t1len);
    
    if (gpu_param) {
        // Use GPU to compute energy history efficiently
        thrust::device_vector<double> d_energy_history(t1len);

        int threads = 64;
        size_t shmem = len_param * sizeof(double) + threads * sizeof(double);

        computeEnergyHistoryKernel<<<t1len, threads, shmem>>>(
            thrust::raw_pointer_cast(simulation.d_QKv.data()),
            thrust::raw_pointer_cast(simulation.d_QRv.data()),
            thrust::raw_pointer_cast(simulation.d_integ.data()),
            thrust::raw_pointer_cast(simulation.d_theta.data()),
            thrust::raw_pointer_cast(simulation.d_t1grid.data()),
            thrust::raw_pointer_cast(d_energy_history.data()),
            T0_param, len_param, t1len
        );
        
        // Copy energy history back to CPU
        thrust::copy(d_energy_history.begin(), d_energy_history.end(), energy_history.begin());
        
        // Extract QK[0] values for each time step
        for (size_t i = 0; i < t1len; ++i) {
            qk0_history[i] = simulation.h_QKv[i * len_param];  // QKv[i * len] is QK[0] at time step i
        }
    } else {
        // CPU computation of energy history
        for (size_t i = 0; i < t1len; ++i) {
            std::vector<double> temp(len_param, 0.0);
            std::vector<double> QKv_i(simulation.h_QKv.begin() + i * len_param, simulation.h_QKv.begin() + (i + 1) * len_param);
            std::vector<double> QRv_i(simulation.h_QRv.begin() + i * len_param, simulation.h_QRv.begin() + (i + 1) * len_param);
            SigmaK(QKv_i, temp);
            energy_history[i] = -(ConvA(temp, QRv_i, simulation.h_t1grid[i])[0] + Dflambda(QKv_i[0]) / T0_param);
            qk0_history[i] = QKv_i[0];  // QK[0] at time step i
        }
    }
    
    // Get directory path from filename
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    
    // Save rvec history
    std::string rvecFilename = dirPath + "/rvec.txt";
    std::ofstream rvecFile(rvecFilename);
    if (rvecFile) {
        rvecFile << std::fixed << std::setprecision(16);
        rvecFile << "# Time\trvec\n";
        for (size_t i = 0; i < t1len; ++i) {
            rvecFile << simulation.h_t1grid[i] << "\t" << simulation.h_rvec[i] << "\n";
        }
        rvecFile.close();
        std::cout << "Saved rvec history to " << rvecFilename << std::endl;
    } else {
        std::cerr << "Error: Could not open file " << rvecFilename << std::endl;
    }
    
    // Save energy history
    std::string energyFilename = dirPath + "/energy.txt";
    std::ofstream energyFile(energyFilename);
    if (energyFile) {
        energyFile << std::fixed << std::setprecision(16);
        energyFile << "# Time\tEnergy\n";
        for (size_t i = 0; i < t1len; ++i) {
            energyFile << simulation.h_t1grid[i] << "\t" << energy_history[i] << "\n";
        }
        energyFile.close();
        std::cout << "Saved energy history to " << energyFilename << std::endl;
    } else {
        std::cerr << "Error: Could not open file " << energyFilename << std::endl;
    }
    
    // Save QK[0] history
    std::string qk0Filename = dirPath + "/qk0.txt";
    std::ofstream qk0File(qk0Filename);
    if (qk0File) {
        qk0File << std::fixed << std::setprecision(16);
        qk0File << "# Time\tQK[0]\n";
        for (size_t i = 0; i < t1len; ++i) {
            qk0File << simulation.h_t1grid[i] << "\t" << qk0_history[i] << "\n";
        }
        qk0File.close();
        std::cout << "Saved QK[0] history to " << qk0Filename << std::endl;
    } else {
        std::cerr << "Error: Could not open file " << qk0Filename << std::endl;
    }
    
    std::cout << "Successfully saved complete history (" << t1len << " time points) to " << dirPath << std::endl;
}

void setupOutputDirectory() {
    // Decide output root based on WHERE THE EXECUTABLE LIVES (not CWD)
    // This restores the original behavior requested by the user.
    char* homeDir = getenv("HOME");
    std::string exePath;
    {
        char exeBuf[PATH_MAX];
        ssize_t n = readlink("/proc/self/exe", exeBuf, sizeof(exeBuf) - 1);
        if (n > 0) { exeBuf[n] = '\0'; exePath = exeBuf; }
    }

    auto canonicalize = [](const std::string& p) -> std::string {
        char buf[PATH_MAX];
        if (!p.empty() && realpath(p.c_str(), buf)) return std::string(buf);
        return p;
    };

    std::string exeCanon = canonicalize(exePath);
    std::string homeCanon = (homeDir ? canonicalize(std::string(homeDir)) : std::string());
    if (!homeCanon.empty() && homeCanon.back() != '/') homeCanon += '/';

    if (!exeCanon.empty() && !homeCanon.empty()) {
        // If the executable resides under canonical HOME, force using outputDir
        if (exeCanon.rfind(homeCanon, 0) == 0) {
            config.resultsDir = config.outputDir;
        }
    }

    std::cout << "Executable (canonical): " << (exeCanon.empty() ? std::string("<unknown>") : exeCanon) << std::endl;
    std::cout << "HOME (canonical): " << (homeCanon.empty() ? std::string("<unknown>") : homeCanon) << std::endl;
    std::cout << "Selected results root: " << config.resultsDir << std::endl;
    
    // Check if directory already exists
    struct stat st = {0};
    if (stat(config.resultsDir.c_str(), &st) == 0) {
        // Directory exists
        if (S_ISDIR(st.st_mode)) {
            std::cout << "Directory already exists: " << config.resultsDir << std::endl;
            // Check if it's writable
            if (config.save_output && access(config.resultsDir.c_str(), W_OK) == 0) {
                std::cout << "Directory is writable." << std::endl;
                return; // Directory exists and is writable, no need to create it
            }
        }
    } else if (config.save_output) {
        // Directory doesn't exist, try to create it
        // For absolute paths, create all parent directories recursively
        if (config.resultsDir[0] == '/') {
            std::string path = "/";
            std::string dirPath = config.resultsDir.substr(1); // Remove leading '/'
            
            // Split the path and create each directory in sequence
            std::istringstream pathStream(dirPath);
            std::string dir;
            
            while (std::getline(pathStream, dir, '/')) {
                if (!dir.empty()) {
                    path += dir + "/";
                    if (stat(path.c_str(), &st) == -1) {
                        if (mkdir(path.c_str(), 0755) != 0) {
                            std::cerr << "Warning: Could not create directory " << path 
                                      << ": " << strerror(errno) << std::endl;
                            // Keep configured resultsDir as-is; later writes may fail, which is preferable to silently changing location
                            break;
                        }
                    }
                }
            }
        } else {
            // Relative path
            if (mkdir(config.resultsDir.c_str(), 0755) != 0) {
                std::cerr << "Warning: Could not create directory " << config.resultsDir 
                          << ": " << strerror(errno) << std::endl;
                // Keep configured resultsDir; do not silently change output location
            } else {
                std::cout << "Created output directory: " << config.resultsDir << std::endl;
            }
        }
    }
}

// Utility functions for save operations
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

double getRuntimeSeconds() {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - program_start_time);
    return duration.count() / 1000.0;
}

std::string formatDuration(double total_seconds) {
    int days = static_cast<int>(total_seconds / 86400);
    int hours = static_cast<int>((total_seconds - days * 86400) / 3600);
    int minutes = static_cast<int>((total_seconds - days * 86400 - hours * 3600) / 60);
    double seconds = total_seconds - days * 86400 - hours * 3600 - minutes * 60;
    
    std::ostringstream oss;
    if (days > 0) {
        oss << days << "d " << hours << "h " << minutes << "m " << std::fixed << std::setprecision(1) << seconds << "s";
    } else if (hours > 0) {
        oss << hours << "h " << minutes << "m " << std::fixed << std::setprecision(1) << seconds << "s";
    } else if (minutes > 0) {
        oss << minutes << "m " << std::fixed << std::setprecision(1) << seconds << "s";
    } else {
        oss << std::fixed << std::setprecision(2) << seconds << "s";
    }
    return oss.str();
}

std::string getGPUInfo() {
    if (!config.gpu) return "None (CPU only)";
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) return "None detected";
    
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::ostringstream oss;
    oss << prop.name << " (Compute " << prop.major << "." << prop.minor << ")";
    return oss.str();
}

std::string formatMemory(size_t memory_kb) {
    if (memory_kb >= 1024 * 1024) {
        return std::to_string(memory_kb / (1024 * 1024)) + " GB";
    } else if (memory_kb >= 1024) {
        return std::to_string(memory_kb / 1024) + " MB";
    } else {
        return std::to_string(memory_kb) + " KB";
    }
}

void saveParametersToFile(const std::string& dirPath, double delta, double delta_t) {
    // Update peak memory before saving
    updatePeakMemory();
    
    std::string filename = dirPath + "/params.txt";
    std::ofstream params(filename);
    if (!params) {
        std::cerr << "Error: Could not open parameter file " << filename << std::endl;
        return;
    }
    
    // Calculate energy
    double energy;
    if (config.gpu) {
        energy = energyGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, config.T0);
    } else {
        vector<double> temp(config.len, 0.0);
        vector<double> lastQKv = getLastLenEntries(sim->h_QKv, config.len);
        SigmaK(lastQKv, temp);
        energy = -(ConvA(temp, getLastLenEntries(sim->h_QRv, config.len), sim->h_t1grid.back())[0] + Dflambda(lastQKv[0])/config.T0);
    }
    
    params << std::setprecision(16);
    params << "# DMFE Simulation Parameters" << std::endl;
    params << "# =========================" << std::endl;
    params << std::endl;

    // VERSION INFORMATION - Add this section first
    params << "# Version Information" << std::endl;
    params << "code_version = " << g_version_info.code_version << std::endl;
    params << "git_hash = " << g_version_info.git_hash << std::endl;
    params << "git_branch = " << g_version_info.git_branch << std::endl;
    if (g_version_info.git_tag != "unknown") {
        params << "git_tag = " << g_version_info.git_tag << std::endl;
    }
    params << "git_dirty = " << (g_version_info.git_dirty ? "true" : "false") << std::endl;
    params << "build_date = " << g_version_info.build_date << std::endl;
    params << "build_time = " << g_version_info.build_time << std::endl;
    params << "compiler_version = " << g_version_info.compiler_version << std::endl;
    params << "cuda_version = " << g_version_info.cuda_version << std::endl;
    params << std::endl;

    // System and Performance Information
    params << "# System Information" << std::endl;
    params << "hostname = " << getHostname() << std::endl;
    params << "timestamp = " << getCurrentTimestamp() << std::endl;
    params << "config.gpu device = " << getGPUInfo() << std::endl;
    params << "execution mode = " << (config.gpu ? "GPU" : "CPU") << std::endl;
    params << std::endl;
    
    params << "# Performance Metrics" << std::endl;
    double runtime_seconds = getRuntimeSeconds();
    params << "runtime seconds = " << std::fixed << std::setprecision(2) << runtime_seconds << std::endl;
    params << "runtime formatted = " << formatDuration(runtime_seconds) << std::endl;
    params << "peak memory usage = " << formatMemory(peak_memory_kb) << std::endl;
    params << "peak memory (kb) = " << peak_memory_kb << std::endl;
    if (config.gpu) {
        params << "peak gpu memory (mb) = " << peak_gpu_memory_mb << std::endl;
        params << "current gpu memory (mb) = " << getGPUMemoryUsage() << std::endl;
    }
    params << "loops per second = " << std::fixed << std::setprecision(2) << (runtime_seconds > 0 ? config.loop / runtime_seconds : 0.0) << std::endl;
    params << std::setprecision(16) << std::defaultfloat;  // Reset again
    params << std::endl;
    
    params << "# Physical Parameters" << std::endl;
    params << "p = " << config.p << std::endl;
    params << "p2 = " << config.p2 << std::endl;
    params << "lambda = " << config.lambda << std::endl;
    // params << "TMCT = " << config.TMCT << std::endl;
    params << "T0 = " << config.T0 << std::endl;
    params << "Gamma = " << config.Gamma << std::endl;
    params << std::endl;
    
    params << "# Numerical Parameters" << std::endl;
    params << "len = " << config.len << std::endl;
    params << "tmax = " << config.tmax << std::endl;
    params << "delta_t_min = " << config.delta_t_min << std::endl;
    params << "delta_max = " << config.delta_max << std::endl;
    params << "maxLoop = " << config.maxLoop << std::endl;
    params << "rmax = [" << config.rmax[0] << ", " << config.rmax[1] << "]" << std::endl;
    params << std::endl;
    
    params << "# Current Simulation State" << std::endl;
    params << "current_time = " << sim->h_t1grid.back() << std::endl;
    params << "current_loop = " << config.loop << std::endl;
    params << "current_delta = " << delta << std::endl;
    params << "current_delta_t = " << delta_t << std::endl;
    params << "current_method = " << (rk->init == 1 ? "RK54" : rk->init == 2 ? "SSPRK104" : "SERK2(" + std::to_string(2 * (rk->init - 2)) + ")") << std::endl;
    params << "current_t1grid_size = " << sim->h_t1grid.size() << std::endl;
    params << "current_QK0 = " << sim->h_QKv[(sim->h_t1grid.size() - 1) * config.len] << std::endl;
    params << "current_QR0 = " << sim->h_QRv[(sim->h_t1grid.size() - 1) * config.len] << std::endl;
    params << "current_r = " << sim->h_rvec.back() << std::endl;
    params << "current_energy = " << energy << std::endl;
    
    params.close();
    std::cout << "Saved parameters to " << filename << std::endl;
}

void saveSimulationStateBinary(const std::string& filename, double delta, double delta_t) {
    // Copy data from GPU to CPU
    if (config.gpu) {
        copyVectorsToCPU(*sim);
    }

    // Calculate energy before saving
    double energy;
    if (config.gpu) {
        energy = energyGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, config.T0);
    } else {
        vector<double> temp(config.len, 0.0);
        vector<double> lastQKv = getLastLenEntries(sim->h_QKv, config.len);
        SigmaK(lastQKv, temp);
        energy = -(ConvA(temp, getLastLenEntries(sim->h_QRv, config.len), sim->h_t1grid.back())[0] + Dflambda(lastQKv[0])/config.T0);
    }
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    // Write header with metadata
    int header_version = 1;
    file.write(reinterpret_cast<char*>(&header_version), sizeof(int));
    
    // Write dimensions
    size_t t1grid_size = sim->h_t1grid.size();
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
    
    // Write time grid
    file.write(reinterpret_cast<const char*>(sim->h_t1grid.data()), sim->h_t1grid.size() * sizeof(double));
    
    // Write vectors
    file.write(reinterpret_cast<const char*>(sim->h_QKv.data()), sim->h_QKv.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(sim->h_QRv.data()), sim->h_QRv.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(sim->h_dQKv.data()), sim->h_dQKv.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(sim->h_dQRv.data()), sim->h_dQRv.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(sim->h_rvec.data()), sim->h_rvec.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(sim->h_drvec.data()), sim->h_drvec.size() * sizeof(double));
    
    std::cout << "Saved binary data to " << filename << std::endl;
    
    // Also write parameters to a separate text file
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    saveParametersToFile(dirPath, delta, delta_t);

    saveHistory(filename, delta, delta_t, *sim, config.len, config.T0, config.gpu);
}

#if defined(H5_RUNTIME_OPTIONAL) || defined(USE_HDF5)
void saveSimulationStateHDF5(const std::string& filename, double delta, double delta_t) {
#if defined(H5_RUNTIME_OPTIONAL)
    if (!h5rt::available()) throw std::runtime_error("HDF5 not available at runtime");
    // Ensure data on CPU
    if (config.gpu) {
        copyVectorsToCPU(*sim);
    }
    // Calculate energy
    double energy;
    if (config.gpu) {
        energy = energyGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, config.T0);
    } else {
        vector<double> temp(config.len, 0.0);
        vector<double> lastQKv = getLastLenEntries(sim->h_QKv, config.len);
        SigmaK(lastQKv, temp);
        energy = -(ConvA(temp, getLastLenEntries(sim->h_QRv, config.len), sim->h_t1grid.back())[0] + Dflambda(lastQKv[0])/config.T0);
    }
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    std::string binFilename = dirPath + "/data.bin";
    if (fileExists(binFilename)) { std::remove(binFilename.c_str()); }

    auto file = h5rt::create_file_trunc(filename.c_str());
    if (file < 0) throw std::runtime_error("Failed to create HDF5 file");

    // Datasets
    h5rt::write_dataset_1d_double(file, "QKv", sim->h_QKv.data(), sim->h_QKv.size());
    h5rt::write_dataset_1d_double(file, "QRv", sim->h_QRv.data(), sim->h_QRv.size());
    h5rt::write_dataset_1d_double(file, "dQKv", sim->h_dQKv.data(), sim->h_dQKv.size());
    h5rt::write_dataset_1d_double(file, "dQRv", sim->h_dQRv.data(), sim->h_dQRv.size());
    h5rt::write_dataset_1d_double(file, "t1grid", sim->h_t1grid.data(), sim->h_t1grid.size());
    h5rt::write_dataset_1d_double(file, "rvec", sim->h_rvec.data(), sim->h_rvec.size());
    h5rt::write_dataset_1d_double(file, "drvec", sim->h_drvec.data(), sim->h_drvec.size());

    // Attributes
    double t_current = sim->h_t1grid.back();
    int current_len = config.len; int current_loop = config.loop;
    h5rt::write_attr_double(file, "time", t_current);
    h5rt::write_attr_int(file, "iteration", current_loop);
    h5rt::write_attr_int(file, "len", current_len);
    h5rt::write_attr_double(file, "delta", config.delta);
    h5rt::write_attr_double(file, "delta_t", config.delta_t);
    h5rt::write_attr_double(file, "T0", config.T0);
    h5rt::write_attr_double(file, "lambda", config.lambda);
    h5rt::write_attr_int(file, "p", config.p);
    h5rt::write_attr_int(file, "p2", config.p2);
    h5rt::write_attr_double(file, "energy", energy);

    h5rt::close_file(file);
    std::cout << "Saved HDF5 data to " << filename << std::endl;
    saveParametersToFile(dirPath, delta, delta_t);
    saveHistory(filename, delta, delta_t, *sim, config.len, config.T0, config.gpu);
#elif defined(USE_HDF5)
    // Copy data from GPU to CPU
    copyVectorsToCPU(*sim);
    
    // Calculate energy before saving
    double energy;
    if (config.gpu) {
        energy = energyGPU(sim->d_QKv, sim->d_QRv, sim->d_t1grid, sim->d_integ, sim->d_theta, config.T0);
    } else {
        vector<double> temp(config.len, 0.0);
        vector<double> lastQKv = getLastLenEntries(sim->h_QKv, config.len);
        SigmaK(lastQKv, temp);
        energy = -(ConvA(temp, getLastLenEntries(sim->h_QRv, config.len), sim->h_t1grid.back())[0] + Dflambda(lastQKv[0])/config.T0);
    }
    
    // Get directory path from filename
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
    
    // Remove any existing binary file to avoid confusion
    std::string binFilename = dirPath + "/data.bin";
    if (fileExists(binFilename)) {
        std::remove(binFilename.c_str());
        std::cout << "Removed existing binary file: " << binFilename << std::endl;
    }
    
    // Create HDF5 file
    H5::H5File file(filename, H5F_ACC_TRUNC);
    
    // Base compression settings
    H5::DSetCreatPropList plist;
    plist.setDeflate(6); // Compression level
    
    // Write QKv dataset with appropriate chunking
    {
        hsize_t qkv_dims[1] = {sim->h_QKv.size()};
        H5::DataSpace qkv_space(1, qkv_dims);
        
        // Set chunk size specifically for this dataset
        size_t chunk_size = std::min(size_t(4096), sim->h_QKv.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        
        file.createDataSet("QKv", H5::PredType::NATIVE_DOUBLE, qkv_space, plist)
            .write(sim->h_QKv.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    // Write QRv dataset with appropriate chunking
    {
        hsize_t qrv_dims[1] = {sim->h_QRv.size()};
        H5::DataSpace qrv_space(1, qrv_dims);
        
        // Set chunk size specifically for this dataset
        size_t chunk_size = std::min(size_t(4096), sim->h_QRv.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        
        file.createDataSet("QRv", H5::PredType::NATIVE_DOUBLE, qrv_space, plist)
            .write(sim->h_QRv.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    // Write dQKv dataset with appropriate chunking
    {
        hsize_t dqkv_dims[1] = {sim->h_dQKv.size()};
        H5::DataSpace dqkv_space(1, dqkv_dims);
        
        // Set chunk size specifically for this dataset
        size_t chunk_size = std::min(size_t(4096), sim->h_dQKv.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        
        file.createDataSet("dQKv", H5::PredType::NATIVE_DOUBLE, dqkv_space, plist)
            .write(sim->h_dQKv.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    // Write dQRv dataset with appropriate chunking
    {
        hsize_t dqrv_dims[1] = {sim->h_dQRv.size()};
        H5::DataSpace dqrv_space(1, dqrv_dims);
        
        // Set chunk size specifically for this dataset
        size_t chunk_size = std::min(size_t(4096), sim->h_dQRv.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        
        file.createDataSet("dQRv", H5::PredType::NATIVE_DOUBLE, dqrv_space, plist)
            .write(sim->h_dQRv.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    // Write t1grid dataset with appropriate chunking
    {
        hsize_t t1_dims[1] = {sim->h_t1grid.size()};
        H5::DataSpace t1_space(1, t1_dims);
        
        // Set chunk size specifically for this dataset - important for smaller arrays!
        size_t chunk_size = std::min(size_t(1024), sim->h_t1grid.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        
        file.createDataSet("t1grid", H5::PredType::NATIVE_DOUBLE, t1_space, plist)
            .write(sim->h_t1grid.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    // Write rvec dataset with appropriate chunking
    {
        hsize_t r_dims[1] = {sim->h_rvec.size()};
        H5::DataSpace r_space(1, r_dims);
        
        // Set chunk size specifically for this dataset - important for smaller arrays!
        size_t chunk_size = std::min(size_t(1024), sim->h_rvec.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        
        file.createDataSet("rvec", H5::PredType::NATIVE_DOUBLE, r_space, plist)
            .write(sim->h_rvec.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    // Write drvec dataset with appropriate chunking
    {
        hsize_t dr_dims[1] = {sim->h_drvec.size()};
        H5::DataSpace dr_space(1, dr_dims);
        
        // Set chunk size specifically for this dataset - important for smaller arrays!
        size_t chunk_size = std::min(size_t(1024), sim->h_drvec.size());
        hsize_t chunk_dims[1] = {chunk_size};
        plist.setChunk(1, chunk_dims);
        
        file.createDataSet("drvec", H5::PredType::NATIVE_DOUBLE, dr_space, plist)
            .write(sim->h_drvec.data(), H5::PredType::NATIVE_DOUBLE);
    }

    // Add metadata as attributes
    H5::DataSpace scalar_space(H5S_SCALAR);
    H5::DataType string_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);
    
    auto add_string_attr = [&file, &scalar_space, &string_type](const char* name, 
                                                               const std::string& value) {
        const char* str_data = value.c_str();
        file.createAttribute(name, string_type, scalar_space)
            .write(string_type, &str_data);
    };
    
    auto add_attr = [&file, &scalar_space](const char* name, 
                                          const H5::PredType& type, 
                                          const void* value) {
        file.createAttribute(name, type, scalar_space)
            .write(type, value);
    };

    // Simulation parameters
    double t_current = sim->h_t1grid.back();
    int current_len = config.len;
    int current_loop = config.loop;

    // Version attributes
    add_string_attr("code_version", g_version_info.code_version);
    add_string_attr("git_hash", g_version_info.git_hash);
    add_string_attr("git_branch", g_version_info.git_branch);
    add_string_attr("build_date", g_version_info.build_date + " " + g_version_info.build_time);
    add_string_attr("compiler", g_version_info.compiler_version);
    add_string_attr("cuda_version", g_version_info.cuda_version);
    
    add_attr("time", H5::PredType::NATIVE_DOUBLE, &t_current);
    add_attr("iteration", H5::PredType::NATIVE_INT, &current_loop);
    add_attr("len", H5::PredType::NATIVE_INT, &current_len);
    add_attr("delta", H5::PredType::NATIVE_DOUBLE, &config.delta);
    add_attr("delta_t", H5::PredType::NATIVE_DOUBLE, &config.delta_t);
    add_attr("T0", H5::PredType::NATIVE_DOUBLE, &config.T0);
    add_attr("lambda", H5::PredType::NATIVE_DOUBLE, &config.lambda);
    add_attr("p", H5::PredType::NATIVE_INT, &config.p);
    add_attr("p2", H5::PredType::NATIVE_INT, &config.p2);
    add_attr("energy", H5::PredType::NATIVE_DOUBLE, &energy);
    
    std::cout << "Saved HDF5 data to " << filename 
              << " (time=" << t_current 
              << ", vectors=" << sim->h_QKv.size() / config.len << "×" << config.len 
              << ", energy=" << energy
              << ")" << std::endl;

    // Save parameter text file directly
    saveParametersToFile(dirPath, delta, delta_t);

    saveHistory(filename, delta, delta_t, *sim, config.len, config.T0, config.gpu);
#endif // inner: H5_RUNTIME_OPTIONAL or USE_HDF5
}
#endif // defined(H5_RUNTIME_OPTIONAL) || defined(USE_HDF5)

void saveSimulationState(const std::string& filename, double delta, double delta_t) {
    std::string dirPath = filename.substr(0, filename.find_last_of('/'));
#if defined(H5_RUNTIME_OPTIONAL)
    if (h5rt::available()) {
        // Use HDF5 if available; else fall through
        try { saveSimulationStateHDF5(filename, delta, delta_t); return; } catch (...) {}
    }
    // Fallback to binary
    saveSimulationStateBinary(dirPath + "/data.bin", delta, delta_t);
#elif defined(USE_HDF5)
    try { saveSimulationStateHDF5(filename, delta, delta_t); }
    catch (...) { std::cerr << "HDF5 error; falling back to binary." << std::endl; saveSimulationStateBinary(dirPath + "/data.bin", delta, delta_t); }
#else
    saveSimulationStateBinary(dirPath + "/data.bin", delta, delta_t);
#endif
}

void saveCompressedData(const std::string& dirPath) {
    // Ensure data is on CPU if using GPU
    if (config.gpu) {
        sim->h_QKB1int.resize(sim->d_QKB1int.size());
        thrust::copy(sim->d_QKB1int.begin(), sim->d_QKB1int.end(), sim->h_QKB1int.begin());
        sim->h_QRB1int.resize(sim->d_QRB1int.size());
        thrust::copy(sim->d_QRB1int.begin(), sim->d_QRB1int.end(), sim->h_QRB1int.begin());
    }
    
    // Save QKB1int to QK_compressed
    std::string qk_filename = dirPath + "/QK_compressed";
    std::ofstream qk_file(qk_filename, std::ios::binary);
    if (!qk_file) {
        std::cerr << "Error: Could not open file " << qk_filename << std::endl;
        return;
    }
    
    // Write header (dimensions information)
    size_t rows = config.len;
    size_t cols = config.len;
    qk_file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    qk_file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    
    // Write data
    qk_file.write(reinterpret_cast<const char*>(sim->h_QKB1int.data()), sim->h_QKB1int.size() * sizeof(double));
    qk_file.close();
    
    // Save QRB1int to QR_compressed
    std::string qr_filename = dirPath + "/QR_compressed";
    std::ofstream qr_file(qr_filename, std::ios::binary);
    if (!qr_file) {
        std::cerr << "Error: Could not open file " << qr_filename << std::endl;
        return;
    }
    
    // Write header (dimensions information)
    qr_file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    qr_file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    
    // Write data
    qr_file.write(reinterpret_cast<const char*>(sim->h_QRB1int.data()), sim->h_QRB1int.size() * sizeof(double));
    qr_file.close();
    
    std::cout << "Saved compressed data to " << qk_filename << " and " << qr_filename << std::endl;
}
