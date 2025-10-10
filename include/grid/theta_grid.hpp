#pragma once

#include <vector>
#include <string>

// Compute the theta grid as defined by the Mathematica construction in the project docs.
// Inputs:
//  - len: grid length (N)
//  - Tmax: maximum time scale (Tmax)
// Output:
//  - theta: filled with 'len' values in [0,1], monotonically increasing
void generate_theta_grid(std::size_t len, double Tmax, std::vector<long double>& theta);

// Compute exact analytical theta value at a given fractional index using high-precision arithmetic.
// This function uses Boost multiprecision (100 digits) to compute the theta value with maximum accuracy.
// Inputs:
//  - idx: fractional index in [0, len-1] (0-based). Can be a real number for interpolation.
//  - len: grid length (N)
//  - Tmax: maximum time scale (Tmax)
// Output:
//  - theta value in [0,1] corresponding to the given fractional index
long double theta_of_index(double idx, std::size_t len, double Tmax);

// Compute exact analytical theta values for a vector of fractional indices (vectorized version).
// This is more efficient than calling theta_of_index repeatedly because it computes the
// Lambert W function and other len/Tmax-dependent quantities only once.
// Inputs:
//  - indices: vector of fractional indices in [0, len-1] (0-based)
//  - len: grid length (N)
//  - Tmax: maximum time scale (Tmax)
// Output:
//  - theta_values: filled with theta values in [0,1] corresponding to the input indices
void theta_of_vec(const std::vector<double>& indices, std::size_t len, double Tmax, 
                  std::vector<long double>& theta_values);

// Write the theta grid to Grid_data/<subdir>/theta.dat, creating the directory if needed.
// Returns the absolute/relative output path written.
std::string write_theta_grid(const std::vector<long double>& theta, std::size_t len, const std::string& subdir);
