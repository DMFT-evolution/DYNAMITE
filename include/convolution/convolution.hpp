#pragma once
#include "core/config_build.hpp"
#include <vector>

// Host convolution functions
std::vector<double> ConvA(const std::vector<double>& f, const std::vector<double>& g, const double t);
std::vector<double> ConvR(const std::vector<double>& f, const std::vector<double>& g, const double t);
