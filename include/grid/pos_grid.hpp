#pragma once

#include <string>
#include <vector>
#include <tuple>

// Generate position grids (posA1y, posA2y, posB2y) from theta and phi grids
// using natural cubic spline interpolation to invert theta with machine precision.
// This implementation matches the Mathematica construction:
//   func = Interpolation[Transpose[{theta[ygrid], ygrid}], Method->"Spline", InterpolationOrder->5]
//   posA1y = func[phi1]; posA2y = func[phi2]; posB2y = func[theta[j] / (phi2 - 10^-200)]
//
// Algorithm:
//   1. Use theta_of_index(i, len, Tmax) to compute exact analytical theta values
//      at densely oversampled points (default 20x, tunable via DMFE_POS_OVERSAMPLE)
//   2. Build natural cubic spline for inverse map: theta -> y using these exact samples
//   3. Evaluate posA1y, posA2y, posB2y using the inverse spline
//
// This achieves high accuracy (~11-12 digits) by using
// exact analytical theta values from generate_theta_grid with arbitrary precision arithmetic,
// combined with C^2 continuous splines. This matches Mathematica's behavior exactly.
// All outputs are row-major N x N matrices of 1-based fractional indices in [1, N].
void generate_pos_grids(std::size_t len,
						double Tmax,
						const std::vector<long double>& theta,
						const std::vector<long double>& phi1,
						const std::vector<long double>& phi2,
						std::vector<double>& posA1y,
						std::vector<double>& posA2y,
						std::vector<double>& posB2y,
						double alpha = 0.0, double delta = 0.0);

// Legacy wrapper: accepts double inputs and promotes to long double internally,
// preserving on-disk double outputs for compatibility.
void generate_pos_grids(std::size_t len,
						double Tmax,
						const std::vector<double>& theta,
						const std::vector<double>& phi1,
						const std::vector<double>& phi2,
						std::vector<double>& posA1y,
						std::vector<double>& posA2y,
						std::vector<double>& posB2y,
						double alpha = 0.0, double delta = 0.0);

// Writing is centralized in grid_io.hpp (write_all_grids)
