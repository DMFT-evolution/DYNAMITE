#include "grid/theta_grid.hpp"
#include "grid/phi_grid.hpp"
#include "grid/pos_grid.hpp"
#include "grid/integration.hpp"
#include "grid/grid_io.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstring>
#include <vector>

static bool ensure_dir(const std::string& path) {
    struct stat st{};
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    if (mkdir(path.c_str(), 0755) == 0) return true;
    return false;
}

std::string write_theta_grid(const std::vector<long double>& theta, std::size_t /*len*/, const std::string& subdir) {
    std::string base = "Grid_data/" + subdir;
    // Create parent and child directories (best-effort)
    ensure_dir("Grid_data");
    ensure_dir(base);
    std::string out = base + "/theta.dat";
    std::ofstream ofs(out);
    if (!ofs) throw std::runtime_error(std::string("Failed to open ") + out + ": " + std::strerror(errno));
    ofs.setf(std::ios::scientific);
    ofs.precision(17);
    for (long double v : theta) ofs << static_cast<double>(v) << '\n';
    return out;
}

// Local helpers formerly spread across files
static void write_matrix_tsv(const std::string& path,
                             const std::vector<long double>& mat,
                             std::size_t N) {
    std::ofstream ofs(path);
    if (!ofs) throw std::runtime_error(std::string("Failed to open ") + path + ": " + std::strerror(errno));
    ofs.setf(std::ios::scientific);
    ofs.precision(17);
    for (std::size_t i = 0; i < N; ++i) {
        const long double* row = &mat[i * N];
        for (std::size_t j = 0; j < N; ++j) {
            if (j) ofs << '\t';
            ofs << row[j];
        }
        ofs << '\n';
    }
}

// Local helpers formerly spread across files
static void write_matrix_tsv(const std::string& path,
                             const std::vector<double>& mat,
                             std::size_t N) {
    std::ofstream ofs(path);
    if (!ofs) throw std::runtime_error(std::string("Failed to open ") + path + ": " + std::strerror(errno));
    ofs.setf(std::ios::scientific);
    ofs.precision(17);
    for (std::size_t i = 0; i < N; ++i) {
        const double* row = &mat[i * N];
        for (std::size_t j = 0; j < N; ++j) {
            if (j) ofs << '\t';
            ofs << row[j];
        }
        ofs << '\n';
    }
}

// Public readers used by validation routines
bool read_vector_file(const std::string& path, std::vector<double>& out) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    out.clear();
    out.reserve(4096);
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        // Trim whitespace
        auto start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        auto end = line.find_last_not_of(" \t\r\n");
        std::string tok = line.substr(start, end - start + 1);
        try {
            out.push_back(std::stod(tok));
        } catch (...) {
            return false;
        }
    }
    return !out.empty();
}

bool read_matrix_tsv(const std::string& path, std::vector<double>& out, std::size_t N) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    out.assign(N * N, 0.0);
    std::string line;
    std::size_t row = 0;
    while (row < N && std::getline(ifs, line)) {
        if (line.empty()) { ++row; continue; }
        std::istringstream iss(line);
        std::size_t col = 0;
        std::string tok;
        while (col < N && std::getline(iss, tok, '\t')) {
            // trim
            auto s = tok.find_first_not_of(" \t\r\n");
            if (s == std::string::npos) { ++col; continue; }
            auto e = tok.find_last_not_of(" \t\r\n");
            std::string t = tok.substr(s, e - s + 1);
            try {
                out[row * N + col] = std::stod(t);
            } catch (...) {
                return false;
            }
            ++col;
        }
        if (col != N) return false; // ensure expected columns
        ++row;
    }
    return row == N;
}

static std::pair<std::string, std::string> write_phi_grids_local(const std::vector<long double>& phi1,
                                                                 const std::vector<long double>& phi2,
                                                                 std::size_t len,
                                                                 const std::string& subdir) {
    std::string base = "Grid_data/" + subdir;
    ensure_dir("Grid_data");
    ensure_dir(base);
    std::string p1 = base + "/phi1.dat";
    std::string p2 = base + "/phi2.dat";
    write_matrix_tsv(p1, phi1, len);
    write_matrix_tsv(p2, phi2, len);
    return {p1, p2};
}

static std::string write_integration_weights_local(const std::vector<long double>& weights,
                                                   std::size_t len,
                                                   const std::string& subdir) {
    std::string base = "Grid_data/" + subdir;
    ensure_dir("Grid_data");
    ensure_dir(base);
    std::string path = base + "/int.dat";
    std::ofstream ofs(path);
    if (!ofs) throw std::runtime_error(std::string("Failed to open ") + path + ": " + std::strerror(errno));
    ofs.setf(std::ios::scientific);
    ofs.precision(17);
    for (std::size_t i = 0; i < len && i < weights.size(); ++i) ofs << static_cast<double>(weights[i]) << '\n';
    return path;
}

static std::tuple<std::string, std::string, std::string>
write_pos_grids_local(const std::vector<double>& posA1y,
                      const std::vector<double>& posA2y,
                      const std::vector<double>& posB2y,
                      std::size_t len,
                      const std::string& subdir) {
    std::string base = "Grid_data/" + subdir;
    ensure_dir("Grid_data");
    ensure_dir(base);
    std::string pA1 = base + "/posA1y.dat";
    std::string pA2 = base + "/posA2y.dat";
    std::string pB2 = base + "/posB2y.dat";
    write_matrix_tsv(pA1, posA1y, len);
    write_matrix_tsv(pA2, posA2y, len);
    write_matrix_tsv(pB2, posB2y, len);
    return {pA1, pA2, pB2};
}

GridPaths write_all_grids(const std::vector<long double>& theta,
                          const std::vector<long double>& phi1,
                          const std::vector<long double>& phi2,
                          const std::vector<long double>& wint,
                          const std::vector<double>& posA1y,
                          const std::vector<double>& posA2y,
                          const std::vector<double>& posB2y,
                          std::size_t len,
                          const std::string& subdir) {
    GridPaths out{};
    out.theta = write_theta_grid(theta, len, subdir);
    auto p = write_phi_grids_local(phi1, phi2, len, subdir);
    out.phi1 = p.first; out.phi2 = p.second;
    out.wint = write_integration_weights_local(wint, len, subdir);
    auto tpos = write_pos_grids_local(posA1y, posA2y, posB2y, len, subdir);
    out.posA1y = std::get<0>(tpos);
    out.posA2y = std::get<1>(tpos);
    out.posB2y = std::get<2>(tpos);
    return out;
}

std::string write_grid_generation_params(std::size_t len,
                                         double Tmax,
                                         int spline_order,
                                         const std::string& interp_method,
                                         int interp_order,
                                         int fh_stencil,
                                         const std::string& subdir,
                                         const std::string& cmdline,
                                         double alpha,
                                         double delta) {
    std::string base = "Grid_data/" + subdir;
    ensure_dir("Grid_data");
    ensure_dir(base);
    std::string path = base + "/grid_params.txt";
    std::ofstream ofs(path);
    if (!ofs) throw std::runtime_error(std::string("Failed to open ") + path + ": " + std::strerror(errno));
    ofs << "# DMFE grid generation parameters\n";
    ofs << "len=" << len << "\n";
    ofs << std::scientific;
    ofs.precision(17);
    ofs << "Tmax=" << Tmax << "\n";
    ofs.unsetf(std::ios::floatfield);
    ofs << "spline_order=" << spline_order << "\n";
    ofs << "interp_method=" << interp_method << "\n";
    ofs << "interp_order=" << interp_order << "\n";
    if (fh_stencil > 0) ofs << "fh_stencil=" << fh_stencil << "\n";
    ofs.setf(std::ios::scientific);
    ofs.precision(17);
    ofs << "alpha=" << alpha << "\n";
    ofs << "delta=" << delta << "\n";
    ofs << "subdir=" << subdir << "\n";
    if (!cmdline.empty()) ofs << "command_line=" << cmdline << "\n";
    return path;
}

std::pair<std::string, std::string>
write_A1_interp_metadata(const std::vector<int>& inds,
                         const std::vector<double>& weights,
                         std::size_t len,
                         const std::string& subdir) {
    std::string base = "Grid_data/" + subdir;
    ensure_dir("Grid_data");
    ensure_dir(base);
    // Write indsA1y.dat as N x N TSV of integer indices
    std::string indsPath = base + "/indsA1y.dat";
    {
        std::ofstream ofs(indsPath);
        if (!ofs) throw std::runtime_error(std::string("Failed to open ") + indsPath + ": " + std::strerror(errno));
        for (std::size_t i = 0; i < len; ++i) {
            const int* row = &inds[i * len];
            for (std::size_t j = 0; j < len; ++j) {
                if (j) ofs << '\t';
                ofs << row[j];
            }
            ofs << '\n';
        }
    }
    // Write weightsA1y.dat as flat vector
    std::string wPath = base + "/weightsA1y.dat";
    {
        std::ofstream ofs(wPath);
        if (!ofs) throw std::runtime_error(std::string("Failed to open ") + wPath + ": " + std::strerror(errno));
        ofs.setf(std::ios::scientific);
        ofs.precision(17);
        for (double v : weights) ofs << v << '\n';
    }
    return {indsPath, wPath};
}

std::pair<std::string, std::string>
write_A2_interp_metadata(const std::vector<int>& inds,
                         const std::vector<double>& weights,
                         std::size_t len,
                         const std::string& subdir) {
    std::string base = "Grid_data/" + subdir;
    ensure_dir("Grid_data");
    ensure_dir(base);
    std::string indsPath = base + "/indsA2y.dat";
    {
        std::ofstream ofs(indsPath);
        if (!ofs) throw std::runtime_error(std::string("Failed to open ") + indsPath + ": " + std::strerror(errno));
        for (std::size_t i = 0; i < len; ++i) {
            const int* row = &inds[i * len];
            for (std::size_t j = 0; j < len; ++j) {
                if (j) ofs << '\t';
                ofs << row[j];
            }
            ofs << '\n';
        }
    }
    std::string wPath = base + "/weightsA2y.dat";
    {
        std::ofstream ofs(wPath);
        if (!ofs) throw std::runtime_error(std::string("Failed to open ") + wPath + ": " + std::strerror(errno));
        ofs.setf(std::ios::scientific);
        ofs.precision(17);
        for (double v : weights) ofs << v << '\n';
    }
    return {indsPath, wPath};
}

std::pair<std::string, std::string>
write_B2_interp_metadata(const std::vector<int>& inds,
                         const std::vector<double>& weights,
                         std::size_t len,
                         const std::string& subdir) {
    std::string base = "Grid_data/" + subdir;
    ensure_dir("Grid_data");
    ensure_dir(base);
    std::string indsPath = base + "/indsB2y.dat";
    {
        std::ofstream ofs(indsPath);
        if (!ofs) throw std::runtime_error(std::string("Failed to open ") + indsPath + ": " + std::strerror(errno));
        for (std::size_t i = 0; i < len; ++i) {
            const int* row = &inds[i * len];
            for (std::size_t j = 0; j < len; ++j) {
                if (j) ofs << '\t';
                ofs << row[j];
            }
            ofs << '\n';
        }
    }
    std::string wPath = base + "/weightsB2y.dat";
    {
        std::ofstream ofs(wPath);
        if (!ofs) throw std::runtime_error(std::string("Failed to open ") + wPath + ": " + std::strerror(errno));
        ofs.setf(std::ios::scientific);
        ofs.precision(17);
        for (double v : weights) ofs << v << '\n';
    }
    return {indsPath, wPath};
}
