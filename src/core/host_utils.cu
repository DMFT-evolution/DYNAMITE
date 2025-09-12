// host_utils.cu
#include "host_utils.hpp"

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
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] * vec2[i];
    }
}

void scaleVec(const vector<double>& vec1, double real, vector<double>& result) {
    if (result.size() != vec1.size()) result.resize(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] * real;
    }
}

void printVectorDifference(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        std::cerr << "Error: Vectors must be of the same length.\n";
        std::cerr << "Size of vector a: " << a.size() << ", Size of vector b: " << b.size() << "\n";
        return;
    }
    double total = 0.0;
    std::cout << "Differences between vectors:\n";
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = std::abs(a[i] - b[i]);
        std::cout << "Index " << i << ": |" << a[i] << " - " << b[i] << "| = " << diff << "\n";
        total += diff;
    }
    std::cout << "Total Differences between vectors: " << total << "\n";
}

