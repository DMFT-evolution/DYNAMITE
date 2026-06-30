#pragma once
#include "core/config_build.hpp"
#include <cstddef>
#include <vector>

// Host interpolation vector functions
void indexVecLN3(const std::vector<double>& weights, const std::vector<size_t>& inds,
                 std::vector<double>& qk_result, std::vector<double>& qr_result, size_t len);

void indexVecN(const size_t length, const std::vector<double>& weights, const std::vector<size_t>& inds, 
               const std::vector<double>& dtratio, std::vector<double>& qK_result, std::vector<double>& qR_result, size_t len);

void indexVecR2(const std::vector<double>& in1, const std::vector<double>& in2, const std::vector<double>& in3, 
                const std::vector<size_t>& inds, const std::vector<double>& dtratio, std::vector<double>& result);

