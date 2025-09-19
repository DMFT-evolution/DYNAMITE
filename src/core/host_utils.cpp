#include "host_utils.hpp"
#include <omp.h>
#include <iostream>
#include <cmath>

using std::vector;

void Product(const vector<double>& vec1, const vector<double>& vec2, vector<double>& result) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must have the same size for component-wise multiplication.");
    }
    if (result.size() != vec1.size()) {
        result.resize(vec1.size());
    }
    #pragma omp parallel for
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] * vec2[i];
    }
}

void Sum(const vector<double>& vec1, const vector<double>& vec2, vector<double>& result) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must have the same size for component-wise multiplication.");
    }

#pragma omp parallel for
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] + vec2[i];
    }
}

void Subtract(const vector<double>& vec1, const vector<double>& vec2, vector<double>& result) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must have the same size for component-wise multiplication.");
    }

#pragma omp parallel for
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] - vec2[i];
    }
}

void scaleVec(const vector<double>& vec1, double real, vector<double>& result) {
    if (result.size() != vec1.size()) result.resize(vec1.size());
    #pragma omp parallel for
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] * real;
    }
}
