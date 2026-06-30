#pragma once
#include "core/config_build.hpp"
#include <vector>
#include <cstddef>

// CPU version of indexMatAll
void indexMatAll(const std::vector<double>& posx, 
                 const std::vector<size_t>& indsy,
                 const std::vector<double>& weightsy, 
                 const std::vector<double>& dtratio,
                 std::vector<double>& qK_result, 
                 std::vector<double>& qR_result);
