// Integration weights on a nonuniform grid using local polynomial (spline-like) integration
// Computes weights w such that sum_i w[i] f(theta[i]) approximates ∫_0^1 f(t) dt
// using a composite rule of degree `order` on each interval [theta[i], theta[i+1]].
// For order=5 (quintic), this matches the reference datasets.

#pragma once
#include <cstddef>
#include <string>
#include <vector>

// Compute integration weights for a given monotone grid theta (size N),
// using a sliding stencil of size (order+1) centered on each interval when possible.
// order must be >=1 and <=8 (practically we use 1,3,5). Default recommended: 5.
void compute_integration_weights(const std::vector<double>& theta,
                                 int order,
                                 std::vector<double>& weights);

// Validate computed weights against Grid_data/<subdir>/int.dat with tolerance tol.
// Returns true if file exists and all differences <= tol; false otherwise.
bool validate_integration_weights(const std::vector<double>& weights,
                                  std::size_t len,
                                  const std::string& subdir,
                                  double tol,
                                  double& maxAbsDiff,
                                  std::size_t& mismatches);
