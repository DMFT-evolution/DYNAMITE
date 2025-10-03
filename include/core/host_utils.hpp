// host_utils.hpp - Small host-side vector helpers used across the project
#pragma once

#include <vector>
// Thrust and CUDA types are not required for these host-only declarations

// Component-wise product: result[i] = vec1[i] * vec2[i]
void Product(const std::vector<double>& vec1, const std::vector<double>& vec2, std::vector<double>& result);
// Scale: result[i] = vec1[i] * real
void scaleVec(const std::vector<double>& vec1, double real, std::vector<double>& result);
// Print per-index absolute differences and their total
void printVectorDifference(const std::vector<double>& a, const std::vector<double>& b);
