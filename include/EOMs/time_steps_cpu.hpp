#ifndef TIME_STEPS_CPU_HPP
#define TIME_STEPS_CPU_HPP

#include <vector>

// Utility functions
std::vector<double> getLastLenEntries(const std::vector<double>& vec,
                                      std::size_t len);

// CPU time-step functions
std::vector<double> QKstep(const std::vector<double>& qK,
                           const std::vector<double>& qR);

std::vector<double> QRstep(const std::vector<double>& qR);

double drstep2(const std::vector<double>& qK,
               const std::vector<double>& qR,
               const std::vector<double>& dqK,
               const std::vector<double>& dqR,
               double t);

// CPU append/replace
void appendAll(const std::vector<double>& qK,
               const std::vector<double>& qR,
               const std::vector<double>& dqK,
               const std::vector<double>& dqR,
               double dr,
               double t);

void replaceAll(const std::vector<double>& qK,
                const std::vector<double>& qR,
                const std::vector<double>& dqK,
                const std::vector<double>& dqR,
                double dr,
                double t);

double energyCPU();

#endif