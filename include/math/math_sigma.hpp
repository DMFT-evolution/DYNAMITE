#pragma once
#include "core/config_build.hpp"
#include <cstddef>
#include <vector>

// Sigma CPU function declarations
void SigmaK(const std::vector<double>& qk, std::vector<double>& result);
void SigmaR(const std::vector<double>& qk, const std::vector<double>& qr, std::vector<double>& result);
std::vector<double> SigmaK10(const std::vector<double>& qk);
std::vector<double> SigmaR10(const std::vector<double>& qk, const std::vector<double>& qr);
std::vector<double> SigmaK01(const std::vector<double>& qk);
std::vector<double> SigmaR01(const std::vector<double>& qk, const std::vector<double>& qr);
