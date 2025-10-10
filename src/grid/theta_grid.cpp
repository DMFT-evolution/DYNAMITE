#include "grid/theta_grid.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/constants/constants.hpp>
#include "math/lambert_w.hpp"

using dmfe::mp;

// Compute exact analytical theta value at fractional index using high-precision arithmetic
// Now exposed as a public function and accepts real (fractional) indices
long double theta_of_index(double idx, std::size_t len, double Tmax) {
    using boost::math::constants::pi;
    const mp mp_len = mp(len);
    const mp mp_Tmax = mp(Tmax);

    // Exact simplification from Mathematica: ProductLog[-1, -π δ] with δ=1/(10 π Tmax) -> ProductLog[-1, -1/(10 Tmax)]
    const mp argW = -mp(1) / (mp(10) * mp_Tmax);
    const mp Wm1 = dmfe::lambertWm1_mp(argW);
    const mp eta = -(mp(2) / mp_len) * Wm1;

    const mp mp_idx = mp(idx + 1.0);  // Convert to 1-based index

    // Use the exact algebraic layout from the Mathematica expression
    const mp t1 = (mp_idx - (mp_len / mp(2)) - mp(1)/mp(2)) * eta;
    const mp tref = (mp(1) - (mp_len / mp(2)) - mp(1)/mp(2)) * eta;
    const mp term1 = (mp(2)/pi<mp>()) * atan(mp(1) / exp(t1));
    const mp term_ref = (mp(2)/pi<mp>()) * atan(mp(1) / exp(tref));
    const mp num = (mp(1) - term1) - (mp(1) - term_ref); // equals term_ref - term1

    const mp thalf = ((mp_len / mp(2)) - mp(1)/mp(2)) * eta;
    const mp den = (mp(2)/pi<mp>()) * atan(exp(thalf)) - (mp(2)/pi<mp>()) * atan(exp(-thalf));

    mp th = num / den;
    long double out = static_cast<long double>(th);
    if (out < 0.0) out = 0.0;
    if (out > 1.0) out = 1.0;
    return out;
}

// Vectorized version: compute theta for multiple indices efficiently
// by computing Wm1 and other len/Tmax-dependent quantities only once
void theta_of_vec(const std::vector<double>& indices, std::size_t len, double Tmax,
                  std::vector<long double>& theta_values) {
    using boost::math::constants::pi;
    const mp mp_len = mp(len);
    const mp mp_Tmax = mp(Tmax);

    // Compute len/Tmax-dependent quantities once
    const mp argW = -mp(1) / (mp(10) * mp_Tmax);
    const mp Wm1 = dmfe::lambertWm1_mp(argW);
    const mp eta = -(mp(2) / mp_len) * Wm1;

    const mp tref = (mp(1) - (mp_len / mp(2)) - mp(1)/mp(2)) * eta;
    const mp term_ref = (mp(2)/pi<mp>()) * atan(mp(1) / exp(tref));

    const mp thalf = ((mp_len / mp(2)) - mp(1)/mp(2)) * eta;
    const mp den = (mp(2)/pi<mp>()) * atan(exp(thalf)) - (mp(2)/pi<mp>()) * atan(exp(-thalf));

    // Reserve output space
    theta_values.resize(indices.size());

    // Compute theta for each index
    for (std::size_t k = 0; k < indices.size(); ++k) {
        const mp mp_idx = mp(indices[k] + 1.0);  // Convert to 1-based index
        
        const mp t1 = (mp_idx - (mp_len / mp(2)) - mp(1)/mp(2)) * eta;
        const mp term1 = (mp(2)/pi<mp>()) * atan(mp(1) / exp(t1));
        const mp num = (mp(1) - term1) - (mp(1) - term_ref);

        mp th = num / den;
        long double out = static_cast<long double>(th);
        if (out < 0.0) out = 0.0;
        if (out > 1.0) out = 1.0;
        theta_values[k] = out;
    }
}

void generate_theta_grid(std::size_t len, double Tmax, std::vector<long double>& theta) {
    // Build indices vector [0, 1, 2, ..., len-1]
    std::vector<double> indices(len);
    for (std::size_t i = 0; i < len; ++i) {
        indices[i] = static_cast<double>(i);
    }
    
    // Use vectorized version for efficiency
    theta_of_vec(indices, len, Tmax, theta);
    
    // Ensure strictly non-decreasing due to numerical noise
    for (std::size_t i = 1; i < len; ++i) {
        if (theta[i] < theta[i-1]) theta[i] = theta[i-1];
    }
}

// write_theta_grid moved to grid_io.cpp
