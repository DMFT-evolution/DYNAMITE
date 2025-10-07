#include "grid/theta_grid.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/constants/constants.hpp>

namespace {
using mp = boost::multiprecision::cpp_dec_float_100;

// High-precision Lambert W, branch -1, for x in [-1/e, 0). Halley's method.
static mp lambertWm1_mp(const mp& x) {
    if (!(x <= mp(0))) throw std::domain_error("lambertWm1 domain x<=0");
    const mp minus_inv_e = -mp(1) / exp(mp(1));
    if (x < minus_inv_e) throw std::domain_error("lambertWm1 domain x>=-1/e");
    if (x == mp(0)) return -std::numeric_limits<double>::infinity();

    mp w;
    if (x > mp(-0.1)) {
        mp L1 = log(-x);
        mp L2 = log(-L1);
        w = L1 - L2 + (L2 / L1);
    } else {
        mp u = x * exp(mp(1)) + mp(1);
        if (u > mp(0)) u = mp(0);
        if (u < mp(-1)) u = mp(-1);
        mp s = sqrt(-mp(2) * u);
        w = -mp(1) - s + (s*s)/mp(3);
    }

    for (int iter = 0; iter < 80; ++iter) {
        mp ew = exp(w);
        mp wew = w * ew;
        mp f = wew - x;
        mp wp1 = w + mp(1);
        mp denom = ew * wp1 - (wp1 + mp(1)) * f / (mp(2) * wp1);
        mp dw = f / denom;
        mp wnext = w - dw;
        if (!std::isfinite(static_cast<double>(wnext))) {
            mp fprime = ew * (w + mp(1));
            dw = f / fprime;
            wnext = w - dw;
        }
        w = wnext;
        if (abs(dw) <= std::numeric_limits<double>::epsilon()) break;
    }
    return w;
}

} // end anonymous namespace

// Compute exact analytical theta value at fractional index using high-precision arithmetic
// Now exposed as a public function and accepts real (fractional) indices
double theta_of_index(double idx, std::size_t len, double Tmax) {
    using boost::math::constants::pi;
    const mp mp_len = mp(len);
    const mp mp_Tmax = mp(Tmax);

    // Exact simplification from Mathematica: ProductLog[-1, -π δ] with δ=1/(10 π Tmax) -> ProductLog[-1, -1/(10 Tmax)]
    const mp argW = -mp(1) / (mp(10) * mp_Tmax);
    const mp Wm1 = lambertWm1_mp(argW);
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
    double out = static_cast<double>(th);
    if (out < 0.0) out = 0.0;
    if (out > 1.0) out = 1.0;
    return out;
}

// Vectorized version: compute theta for multiple indices efficiently
// by computing Wm1 and other len/Tmax-dependent quantities only once
void theta_of_vec(const std::vector<double>& indices, std::size_t len, double Tmax,
                  std::vector<double>& theta_values) {
    using boost::math::constants::pi;
    const mp mp_len = mp(len);
    const mp mp_Tmax = mp(Tmax);

    // Compute len/Tmax-dependent quantities once
    const mp argW = -mp(1) / (mp(10) * mp_Tmax);
    const mp Wm1 = lambertWm1_mp(argW);
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
        double out = static_cast<double>(th);
        if (out < 0.0) out = 0.0;
        if (out > 1.0) out = 1.0;
        theta_values[k] = out;
    }
}

void generate_theta_grid(std::size_t len, double Tmax, std::vector<double>& theta) {
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
