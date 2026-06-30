#ifndef SEARCH_UTILS_HPP
#define SEARCH_UTILS_HPP

#include "core/config_build.hpp"
#include <vector>


// CPU binary search function
std::vector<double> bsearchPosSorted(const std::vector<double>& list, const std::vector<double>& elem);

// CPU interpolation search with initial values
std::vector<double> isearchPosSortedInit(const std::vector<double>& list, const std::vector<double>& elem, const std::vector<double>& inits);

#endif // SEARCH_UTILS_HPP
