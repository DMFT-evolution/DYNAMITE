#include "io/io_utils.hpp"
#include "simulation/simulation_data.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#if DMFE_WITH_CUDA
#include <thrust/copy.h>
#endif
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

extern SimulationConfig config;
extern SimulationData* sim;

void saveCompressedData(const std::string& dirPath)
{
#if DMFE_WITH_CUDA
    if (config.gpu) {
        sim->h_QKB1int.resize(sim->d_QKB1int.size()); thrust::copy(sim->d_QKB1int.begin(), sim->d_QKB1int.end(), sim->h_QKB1int.begin());
        sim->h_QRB1int.resize(sim->d_QRB1int.size()); thrust::copy(sim->d_QRB1int.begin(), sim->d_QRB1int.end(), sim->h_QRB1int.begin());
        sim->h_theta.resize(sim->d_theta.size()); thrust::copy(sim->d_theta.begin(), sim->d_theta.end(), sim->h_theta.begin());
        sim->h_t1grid.resize(sim->d_t1grid.size()); thrust::copy(sim->d_t1grid.begin(), sim->d_t1grid.end(), sim->h_t1grid.begin());
    }
#endif

    const double p_start_cmp = 0.80, p_end_cmp = 0.90, p_span_cmp = (p_end_cmp - p_start_cmp);
    const double last_t1_cmp = sim->h_t1grid.empty() ? 0.0 : sim->h_t1grid.back();
    auto update_cmp_prog = [&](size_t done, size_t total){ double frac = p_start_cmp + (total ? (p_span_cmp * (double)done / (double)total) : 0.0); if (frac > p_end_cmp) frac = p_end_cmp; if (frac < p_start_cmp) frac = p_start_cmp; _setSaveProgress(frac, last_t1_cmp, "compressed"); };
    size_t total_bytes = sizeof(size_t)*4 + sim->h_QKB1int.size()*sizeof(double) + sim->h_QRB1int.size()*sizeof(double) + sim->h_theta.size()*24;
    size_t done_bytes = 0; update_cmp_prog(done_bytes, total_bytes);

    // QK
    std::string qk_filename = dirPath + "/QK_compressed";
    std::ofstream qk_file(qk_filename, std::ios::binary);
    if (!qk_file) { std::cerr << dmfe::console::ERR() << "Could not open file " << qk_filename << std::endl; return; }
    size_t rows = config.len, cols = config.len;
    qk_file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t)); done_bytes += sizeof(size_t);
    qk_file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t)); done_bytes += sizeof(size_t); update_cmp_prog(done_bytes, total_bytes);
    qk_file.write(reinterpret_cast<const char*>(sim->h_QKB1int.data()), sim->h_QKB1int.size() * sizeof(double)); done_bytes += sim->h_QKB1int.size() * sizeof(double); update_cmp_prog(done_bytes, total_bytes); qk_file.close();

    // QR
    std::string qr_filename = dirPath + "/QR_compressed";
    std::ofstream qr_file(qr_filename, std::ios::binary);
    if (!qr_file) { std::cerr << dmfe::console::ERR() << "Could not open file " << qr_filename << std::endl; return; }
    qr_file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t)); done_bytes += sizeof(size_t);
    qr_file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t)); done_bytes += sizeof(size_t); update_cmp_prog(done_bytes, total_bytes);
    qr_file.write(reinterpret_cast<const char*>(sim->h_QRB1int.data()), sim->h_QRB1int.size() * sizeof(double)); done_bytes += sim->h_QRB1int.size() * sizeof(double); update_cmp_prog(done_bytes, total_bytes); qr_file.close();

    // t1 * theta
    double last_t1 = sim->h_t1grid.back();
    std::string t1_filename = dirPath + "/t1_compressed.txt";
    std::ofstream t1_file(t1_filename);
    if (!t1_file) { std::cerr << dmfe::console::ERR() << "Could not open file " << t1_filename << std::endl; return; }
    t1_file << std::fixed << std::setprecision(16);
    for (size_t i = 0; i < sim->h_theta.size(); ++i) {
        t1_file << last_t1 * sim->h_theta[i] << "\n";
        done_bytes += 24; if ((i & 0x3FF) == 0) update_cmp_prog(done_bytes, total_bytes);
    }
    t1_file.close();

    if (config.debug) {
        std::cout << dmfe::console::SAVE() << "Saved compressed data to " << qk_filename << ", " << qr_filename << ", and " << t1_filename << std::endl;
    }
    update_cmp_prog(total_bytes, total_bytes);
}

void saveCompressedDataAsync(const std::string& dirPath, const SimulationDataSnapshot& snapshot)
{
    const double p_start_cmp = 0.80, p_end_cmp = 0.90, p_span_cmp = (p_end_cmp - p_start_cmp);
    const double last_t1_cmp = snapshot.t1grid.empty() ? 0.0 : snapshot.t1grid.back();
    auto update_cmp_prog = [&](size_t done, size_t total){ double frac = p_start_cmp + (total ? (p_span_cmp * (double)done / (double)total) : 0.0); if (frac > p_end_cmp) frac = p_end_cmp; if (frac < p_start_cmp) frac = p_start_cmp; _setSaveProgress(frac, last_t1_cmp, "compressed"); };
    size_t total_bytes = sizeof(size_t)*4 + snapshot.QKB1int.size()*sizeof(double) + snapshot.QRB1int.size()*sizeof(double) + snapshot.theta.size()*24;
    size_t done_bytes = 0; update_cmp_prog(done_bytes, total_bytes);

    std::string qk_filename = dirPath + "/QK_compressed";
    std::ofstream qk_file(qk_filename, std::ios::binary);
    if (!qk_file) { std::cerr << dmfe::console::ERR() << "Could not open file " << qk_filename << std::endl; return; }
    size_t rows = snapshot.current_len, cols = snapshot.current_len;
    qk_file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t)); done_bytes += sizeof(size_t);
    qk_file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t)); done_bytes += sizeof(size_t); update_cmp_prog(done_bytes, total_bytes);
    qk_file.write(reinterpret_cast<const char*>(snapshot.QKB1int.data()), snapshot.QKB1int.size() * sizeof(double)); done_bytes += snapshot.QKB1int.size() * sizeof(double); update_cmp_prog(done_bytes, total_bytes); qk_file.close();

    std::string qr_filename = dirPath + "/QR_compressed";
    std::ofstream qr_file(qr_filename, std::ios::binary);
    if (!qr_file) { std::cerr << dmfe::console::ERR() << "Could not open file " << qr_filename << std::endl; return; }
    qr_file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t)); done_bytes += sizeof(size_t);
    qr_file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t)); done_bytes += sizeof(size_t); update_cmp_prog(done_bytes, total_bytes);
    qr_file.write(reinterpret_cast<const char*>(snapshot.QRB1int.data()), snapshot.QRB1int.size() * sizeof(double)); done_bytes += snapshot.QRB1int.size() * sizeof(double); update_cmp_prog(done_bytes, total_bytes); qr_file.close();

    double last_t1 = snapshot.t1grid.back();
    std::string t1_filename = dirPath + "/t1_compressed.txt";
    std::ofstream t1_file(t1_filename);
    if (!t1_file) { std::cerr << dmfe::console::ERR() << "Could not open file " << t1_filename << std::endl; return; }
    t1_file << std::fixed << std::setprecision(16);
    for (size_t i = 0; i < snapshot.theta.size(); ++i) {
        t1_file << last_t1 * snapshot.theta[i] << "\n";
        done_bytes += 24; if ((i & 0x3FF) == 0) update_cmp_prog(done_bytes, total_bytes);
    }
    t1_file.close();

    if (config.debug) {
        std::cout << dmfe::console::SAVE() << "Saved compressed data to " << qk_filename << ", " << qr_filename << ", and " << t1_filename << " (async)" << std::endl;
    }
    update_cmp_prog(total_bytes, total_bytes);
}
