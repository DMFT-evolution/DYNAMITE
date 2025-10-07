#pragma once

#include <string>
#include <vector>

// Generate phi1 and phi2 grids from a 1D theta grid.
// Inputs:
//  - theta: length-N vector in [0,1]
// Outputs (row-major N x N):
//  - phi1[i*N + j] = theta[i] * theta[j]
//  - phi2[i*N + j] = theta[i] + (1 - theta[i]) * theta[j]
void generate_phi_grids(const std::vector<double>& theta,
						std::vector<double>& phi1,
						std::vector<double>& phi2);

// If reference files exist in Grid_data/<subdir>, compare element-wise with tolerance.
// Returns true if files existed and all differences <= tol; false if files missing or mismatch.
// On return, maxAbsDiff has the maximum absolute difference seen (or 0 if files missing), and
// mismatches counts values with abs diff > tol.
bool validate_against_saved(const std::vector<double>& theta,
							const std::vector<double>& phi1,
							const std::vector<double>& phi2,
							std::size_t len,
							const std::string& subdir,
							double tol,
							double& maxAbsDiff,
							std::size_t& mismatches);
