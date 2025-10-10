// Spline-consistent integration weights on a nonuniform grid.
// Computes weights w such that sum_i w[i] f(theta[i]) equals the integral of the
// degree-`order` open-clamped interpolatory B-spline s that interpolates samples f(theta[i]).
// This matches Mathematica's cubic/quintic behavior and generalizes to arbitrary degree.

#pragma once
#include <cstddef>
#include <string>
#include <vector>

// Compute integration weights for a given monotone grid theta (size N), using a global
// degree-`order` open-clamped B-spline collocation. order is the spline degree (>=1),
// and is clamped to [1, N-1]. Preferred high-precision API: theta as long double.
void compute_integration_weights(const std::vector<long double>& theta,
                                 int order,
                                 std::vector<long double>& weights);

// Backward-compatible overload for double theta (casts to long double internally)
void compute_integration_weights(const std::vector<double>& theta,
                                 int order,
                                 std::vector<long double>& weights);

// Validate computed weights against Grid_data/<subdir>/int.dat with tolerance tol.
// Returns true if file exists and all differences <= tol; false otherwise.
bool validate_integration_weights(const std::vector<double>& weights,
                                  std::size_t len,
                                  const std::string& subdir,
                                  double tol,
                                  double& maxAbsDiff,
                                  std::size_t& mismatches);
