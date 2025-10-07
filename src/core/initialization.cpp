#include "core/initialization.hpp"
#include "core/config.hpp"
#include "core/config_build.hpp"
#include "simulation/simulation_data.hpp"
#include "EOMs/rk_data.hpp"
#if DMFE_WITH_CUDA
#include "core/device_utils.cuh"
#include "core/gpu_memory_utils.hpp"
#endif
#include "io/io_utils.hpp"
#include "math/math_ops.hpp"
#include "EOMs/time_steps.hpp"
#include "EOMs/runge_kutta.hpp"
#include "interpolation/interpolation_core.hpp"
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#include <dirent.h>
#include <cstdio>

// External global objects (defined in main executable)
extern SimulationConfig config;
extern SimulationData* sim;
extern RKData* rk;

void init()
{
    // Helper: does a regular file exist?
    auto file_exists = [&](const std::string& path) -> bool {
        struct stat st{}; return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
    };

    // Helper: ensure Grid_data/<len>/ exists with required artifacts; otherwise generate it via CLI
    auto ensure_grids_for_length = [&](std::size_t L) {
        const std::string baseDir = std::string("Grid_data/") + std::to_string(L);
        const std::string theta   = baseDir + "/theta.dat";
        const std::string phi1    = baseDir + "/phi1.dat";
        const std::string phi2    = baseDir + "/phi2.dat";
        const std::string wint    = baseDir + "/int.dat";
        const std::string indsA1  = baseDir + "/indsA1y.dat";
        const std::string wA1     = baseDir + "/weightsA1y.dat";
        const std::string indsA2  = baseDir + "/indsA2y.dat";
        const std::string wA2     = baseDir + "/weightsA2y.dat";
        const std::string indsB2  = baseDir + "/indsB2y.dat";
        const std::string wB2     = baseDir + "/weightsB2y.dat";
        const std::string posA1   = baseDir + "/posA1y.dat";
        const std::string posA2   = baseDir + "/posA2y.dat";
        const std::string posB2   = baseDir + "/posB2y.dat";

        auto grids_present = [&](){
            return file_exists(theta) && file_exists(phi1) && file_exists(phi2) && file_exists(wint)
                && file_exists(indsA1) && file_exists(wA1)
                && file_exists(indsA2) && file_exists(wA2)
                && file_exists(indsB2) && file_exists(wB2)
                && file_exists(posA1) && file_exists(posA2) && file_exists(posB2);
        };

        if (grids_present()) return; // Nothing to do

        // Try to reuse any existing Grid_data subdir that matches requested length
        auto is_dir = [](const std::string& p){ struct stat st{}; return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode); };
        auto read_len_from_params = [&](const std::string& paramsPath) -> long long {
            std::ifstream in(paramsPath);
            if (!in) return -1;
            std::string line;
            while (std::getline(in, line)) {
                if (line.rfind("len=", 0) == 0) {
                    try { return std::stoll(line.substr(4)); } catch (...) { return -1; }
                }
            }
            return -1;
        };
        auto copy_file = [&](const std::string& src, const std::string& dst){
            std::ifstream i(src, std::ios::binary);
            std::ofstream o(dst, std::ios::binary);
            o << i.rdbuf();
            return (bool)o;
        };

        const std::string root = "Grid_data";
        if (is_dir(root)) {
            DIR* d = ::opendir(root.c_str());
            if (d) {
                struct dirent* ent;
                while ((ent = ::readdir(d)) != nullptr) {
                    if (ent->d_name[0] == '.') continue;
                    std::string sub = root + "/" + ent->d_name;
                    if (!is_dir(sub)) continue;
                    const std::string params = sub + "/grid_params.txt";
                    long long Lp = read_len_from_params(params);
                    if (Lp == (long long)L) {
                        // Candidate directory has matching len and all required files
                        auto present_in_sub = [&](){
                            auto req = {"theta.dat","phi1.dat","phi2.dat","int.dat",
                                        "indsA1y.dat","weightsA1y.dat","indsA2y.dat","weightsA2y.dat",
                                        "indsB2y.dat","weightsB2y.dat","posA1y.dat","posA2y.dat","posB2y.dat"};
                            for (auto&& f : req) if (!file_exists(sub + "/" + f)) return false; return true;
                        };
                        if (present_in_sub()) {
                            if (sub == baseDir) return; // exact path exists logically
                            // Create symlink Grid_data/<L> -> sub (if not already a dir)
                            struct stat st{};
                            if (::lstat(baseDir.c_str(), &st) == 0) {
                                // remove existing file/symlink to replace
                                ::unlink(baseDir.c_str());
                            }
                            if (::symlink(ent->d_name, (root + "/" + std::to_string(L)).c_str()) == 0) {
                                std::cout << "Using existing grids at " << sub << " via symlink " << baseDir << std::endl;
                                if (grids_present()) { ::closedir(d); return; }
                            }
                            // Fallback: copy files
                            ::mkdir(baseDir.c_str(), 0755);
                            bool ok = true;
                            ok &= copy_file(sub + "/theta.dat", theta);
                            ok &= copy_file(sub + "/phi1.dat",  phi1);
                            ok &= copy_file(sub + "/phi2.dat",  phi2);
                            ok &= copy_file(sub + "/int.dat",   wint);
                            ok &= copy_file(sub + "/indsA1y.dat", indsA1);
                            ok &= copy_file(sub + "/weightsA1y.dat", wA1);
                            ok &= copy_file(sub + "/indsA2y.dat", indsA2);
                            ok &= copy_file(sub + "/weightsA2y.dat", wA2);
                            ok &= copy_file(sub + "/indsB2y.dat", indsB2);
                            ok &= copy_file(sub + "/weightsB2y.dat", wB2);
                            ok &= copy_file(sub + "/posA1y.dat", posA1);
                            ok &= copy_file(sub + "/posA2y.dat", posA2);
                            ok &= copy_file(sub + "/posB2y.dat", posB2);
                            if (ok && grids_present()) { ::closedir(d); return; }
                        }
                    }
                }
                ::closedir(d);
            }
        }

        std::cout << "No existing grids found for len=" << L << ", generating defaults via CLI..." << std::endl;

        // Resolve current executable path (Linux: /proc/self/exe)
        char exePath[PATH_MAX]; ssize_t n = ::readlink("/proc/self/exe", exePath, sizeof(exePath)-1);
        std::string self = (n > 0 ? (exePath[n] = '\0', std::string(exePath)) : std::string("./RG-Evo"));

        // Build command: <self> grid -L <L>
        std::ostringstream cmd;
        cmd << '"' << self << '"' << " grid -L " << L;

    int rc = std::system(cmd.str().c_str());
        if (rc != 0 || !grids_present()) {
            std::cerr << "Error: failed to generate grids for len=" << L << ". Exit code=" << rc << std::endl;
            throw std::runtime_error("grid generation failed");
        }
        std::cout << "Grids generated at " << baseDir << std::endl;
    };

    // Handle GPU configuration based on user preference and hardware availability
#if DMFE_WITH_CUDA
    if (config.gpu) {
        if (isCompatibleGPUInstalled()) {
            std::cout << "GPU acceleration enabled." << std::endl;
            config.gpu = true;
        } else {
            std::cout << "Warning: GPU acceleration requested but no compatible GPU found. Falling back to CPU." << std::endl;
            config.gpu = false;
        }
    } else {
        std::cout << "GPU acceleration disabled by user. Using CPU." << std::endl;
        config.gpu = false;
    }
#else
    std::cout << "Running in CPU-only mode." << std::endl;
    config.gpu = false;
#endif
    
    sim = new SimulationData();
    rk = new RKData();

    setupOutputDirectory();

    // Ensure required grids for the requested length exist (or generate with defaults)
    ensure_grids_for_length(config.len);

    // Load grids
    import(*sim, config.len, config.ord);

    // Generate filename based on parameters
    std::string filename = getFilename(config.resultsDir, config.p, config.p2, config.lambda, config.T0, config.Gamma, config.len, config.delta_t_min, config.delta_max, config.use_serk2, config.aggressive_sparsify, config.save_output);
    bool loaded = false;
    LoadedStateParams loaded_params;  // Declare the structure to hold loaded parameters

    // Try to load existing simulation data
    if (fileExists(filename) || fileExists(filename.substr(0, filename.find_last_of('.')) + ".bin")) {
        std::cout << "Found existing simulation file. Attempting to load..." << std::endl;
        loaded = loadSimulationState(filename, *sim, config.p, config.p2, config.lambda, config.T0, config.Gamma, config.len, config.delta_t_min, config.delta_max, config.use_serk2, config.aggressive_sparsify, loaded_params);
        if (loaded) {
            config.loaded = true;
            std::string dirPath = filename.substr(0, filename.find_last_of('/'));
            config.paramDir = dirPath;
        }
    }
    
    if (!loaded) {
        // Start new simulation with default values
        std::cout << "New simulation..." << std::endl;
        
        sim->h_t1grid.resize(1, 0.0);
        sim->h_delta_t_ratio.resize(1, 0.0);
        config.specRad = 4 * sqrt(DDflambda(1));

        config.delta_t = config.delta_t_min;
        config.loop = 0;
        config.delta = 1;
        config.delta_old = 0;

        sim->h_QKv.resize(config.len, 1.0);
        sim->h_QRv.resize(config.len, 1.0);
        sim->h_dQKv.resize(config.len, 0.0);
        sim->h_dQRv.resize(config.len, 0.0);
        sim->h_rvec.resize(1, config.Gamma + Dflambda(1) / config.T0);
        sim->h_drvec.resize(1, rstep());
    } else {
        // We successfully loaded data, use the loaded parameters
        config.delta = loaded_params.delta;
        config.delta_t = loaded_params.delta_t;
        config.loop = loaded_params.loop;
        config.specRad = 4 * sqrt(DDflambda(1));
        std::cout << "Loaded simulation state: delta=" << config.delta 
                  << ", delta_t=" << config.delta_t << ", loop=" << config.loop << std::endl;
    }

    // Initialize intermediate arrays needed for interpolation (always needed)
    sim->h_posB1xOld.resize(config.len, 1.0);
    sim->h_posB2xOld.resize(config.len * config.len, 0.0);

    sim->h_SigmaKA1int.resize(config.len * config.len, 0.0);
    sim->h_SigmaRA1int.resize(config.len * config.len, 0.0);
    sim->h_SigmaKB1int.resize(config.len * config.len, 0.0);
    sim->h_SigmaRB1int.resize(config.len * config.len, 0.0);
    sim->h_SigmaKA2int.resize(config.len * config.len, 0.0);
    sim->h_SigmaRA2int.resize(config.len * config.len, 0.0);
    sim->h_SigmaKB2int.resize(config.len * config.len, 0.0);
    sim->h_SigmaRB2int.resize(config.len * config.len, 0.0);

    sim->h_QKA1int.resize(config.len * config.len, 0.0);
    sim->h_QRA1int.resize(config.len * config.len, 0.0);
    sim->h_QKB1int.resize(config.len * config.len, 0.0);
    sim->h_QRB1int.resize(config.len * config.len, 0.0);
    sim->h_QKA2int.resize(config.len * config.len, 0.0);
    sim->h_QRA2int.resize(config.len * config.len, 0.0);
    sim->h_QKB2int.resize(config.len * config.len, 0.0);
    sim->h_QRB2int.resize(config.len * config.len, 0.0);

    sim->h_rInt.resize(config.len, 0.0);
    sim->h_drInt.resize(config.len, 0.0);

    if (config.gpu) {
#if DMFE_WITH_CUDA
        copyVectorsToGPU(*sim, config.len);
        copyParametersToDevice(config.p, config.p2, config.lambda);

        // (IndexVecLN3 optimizer setup no longer needed here; handled inside indexVecLN3GPU)
        // weightsB2y optimization setup now handled internally in indexMatAllGPU (if needed)

        // Initialize the appropriate GPU RK method to ensure required buffers are set
        if (config.delta_t < config.rmax[0] / config.specRad) {
            init_RK54GPU();   // RK54
        } else {
            init_SSPRK104GPU(); // SSPRK104
        }

        // Initial interpolation on GPU
        interpolateGPU();
#endif
    } else {
        // Choose CPU RK method consistent with previous version's behavior
        if (config.delta_t < config.rmax[0] / config.specRad) {
            rk->init = 1;  // RK54
        } else {
            rk->init = 2;  // SSPRK104
        }
        interpolate();
    }
}
