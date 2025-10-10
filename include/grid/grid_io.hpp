#pragma once

#include <cstddef>
#include <string>
#include <vector>

// Paths returned by unified grid writer
struct GridPaths {
    std::string theta;
    std::string phi1;
    std::string phi2;
    std::string wint;    // integration weights (int.dat)
    std::string posA1y;
    std::string posA2y;
    std::string posB2y;
};

// Write all grid artifacts into Grid_data/<subdir>:
// - theta.dat (vector of size N)
// - phi1.dat, phi2.dat (N x N TSV)
// - int.dat (vector of size N)
// - posA1y.dat, posA2y.dat, posB2y.dat (N x N TSV)
// Returns the file paths.
GridPaths write_all_grids(const std::vector<long double>& theta,
                          const std::vector<long double>& phi1,
                          const std::vector<long double>& phi2,
                          const std::vector<long double>& wint,
                          const std::vector<double>& posA1y,
                          const std::vector<double>& posA2y,
                          const std::vector<double>& posB2y,
                          std::size_t len,
                          const std::string& subdir);

// Write a small text file capturing the parameters used to generate the grid
// package into Grid_data/<subdir>/grid_params.txt. Returns the file path.
// If fh_stencil <= 0 it is omitted (defaults to n+1).
std::string write_grid_generation_params(std::size_t len,
                                         double Tmax,
                                         int spline_order,
                                         const std::string& interp_method,
                                         int interp_order,
                                         int fh_stencil,
                                         const std::string& subdir,
                                         const std::string& cmdline);

// Centralized grid readers (used by validation code)
// Reads a whitespace-separated vector file into 'out'; returns true on success with non-empty data
bool read_vector_file(const std::string& path, std::vector<double>& out);

// Reads an N x N TSV matrix into 'out' (row-major, size N*N); returns true on success and complete read
bool read_matrix_tsv(const std::string& path, std::vector<double>& out, std::size_t N);

// Write interpolation metadata for A1 (phi1 as target):
// - indsA1y.dat: N x N TSV matrix of start indices (first contributing node) per entry
// - weightsA1y.dat: flat vector of weights concatenated per entry
// Returns pair of output paths {indsPath, weightsPath}.
std::pair<std::string, std::string>
write_A1_interp_metadata(const std::vector<int>& inds,
                         const std::vector<double>& weights,
                         std::size_t len,
                         const std::string& subdir);

// Write interpolation metadata for A2 (phi2 as target)
std::pair<std::string, std::string>
write_A2_interp_metadata(const std::vector<int>& inds,
                         const std::vector<double>& weights,
                         std::size_t len,
                         const std::string& subdir);

// Write interpolation metadata for B2 (theta/(phi2-eps) as target)
std::pair<std::string, std::string>
write_B2_interp_metadata(const std::vector<int>& inds,
                         const std::vector<double>& weights,
                         std::size_t len,
                         const std::string& subdir);
