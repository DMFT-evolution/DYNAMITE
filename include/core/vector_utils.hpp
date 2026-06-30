#pragma once
#include "core/config_build.hpp"
#include <vector>
#include <stdexcept>
#include <algorithm>

// Host vector operators
inline std::vector<double>& operator+=(std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) throw std::invalid_argument("Vectors must be same size for +=");
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), std::plus<double>());
    return lhs;
}
inline std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors must match for +");
    std::vector<double> r(a.size());
    std::transform(a.begin(), a.end(), b.begin(), r.begin(), std::plus<>());
    return r;
}
inline std::vector<double> operator-(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors must match for -");
    std::vector<double> r(a.size());
    std::transform(a.begin(), a.end(), b.begin(), r.begin(), std::minus<>());
    return r;
}
inline std::vector<double> operator*(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors must match for * element-wise");
    std::vector<double> r(a.size());
    std::transform(a.begin(), a.end(), b.begin(), r.begin(), std::multiplies<>());
    return r;
}
inline std::vector<double> operator*(const std::vector<double>& a, double s) {
    std::vector<double> r(a.size());
    std::transform(a.begin(), a.end(), r.begin(), [s](double v){return v*s;});
    return r;
}
inline std::vector<double> operator*(double s, const std::vector<double>& a) { return a * s; }
