#include "grid/theta_grid.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "math/mp_type.hpp"
#include "math/lambert_w.hpp"

using dmfe::mp;

// Helper: non-linear index remapping per Mathematica ThetaFunc2 transform.
// Maps x in [1, len] (1-based fractional index) to x' in [1, len].
static inline long double remap_index(long double x1based, std::size_t len, double alpha, double delta) {
    if (alpha == 0.0) return x1based; // identity fast-path
    const long double N = static_cast<long double>(len);
    const long double c = (N - 1.0L) / 2.0L;
    const long double t = 2.0L * x1based - N - 1.0L;     // in [- (N+1), +(N-1)] for x in [1,N]
    const long double denom = (N - 1.0L);
    long double s = t / denom;                            // normalized symmetric coordinate in [-1,1]
    if (s > 1.0L) s = 1.0L; if (s < -1.0L) s = -1.0L;
    const long double abs_s = std::fabs(s);
    const long double d = static_cast<long double>(delta);
    // Smooth odd transform: Sign[s] * ((|s|^3 + d^3)^(1/3) - d)
    const long double inner = std::pow(abs_s * abs_s * abs_s + d * d * d, 1.0L/3.0L) - d;
    const long double S = (s < 0 ? -1.0L : (s > 0 ? 1.0L : 0.0L));
    const long double g = S * inner;
    const long double g1 = (std::pow(1.0L + d*d*d, 1.0L/3.0L) - d); // normalization at |s|=1
    long double mapped = c * ( (g / g1) + 1.0L ) + 1.0L; // in [1,N]
    // Blend with identity via alpha
    long double out = static_cast<long double>(alpha) * mapped + (1.0L - static_cast<long double>(alpha)) * x1based;
    // Clamp to [1,N]
    if (out < 1.0L) out = 1.0L; if (out > N) out = N;
    return out;
}

// Compute exact analytical theta value at fractional index using high-precision arithmetic
// Now exposed as a public function and accepts real (fractional) indices
long double theta_of_index(double idx, std::size_t len, double Tmax, double alpha, double delta) {
    const mp pi = dmfe::mp_pi();
    const mp mp_len = mp(len);
    const mp mp_Tmax = mp(Tmax);

    // Exact simplification from Mathematica: ProductLog[-1, -π δ] with δ=1/(10 π Tmax) -> ProductLog[-1, -1/(10 Tmax)]
    const mp argW = -mp(1) / (mp(10) * mp_Tmax);
    const mp Wm1 = dmfe::lambertWm1_mp(argW);
    const mp eta = -(mp(2) / mp_len) * Wm1;

    // Map to 1-based fractional index and apply optional non-linear remapping
    long double x1 = static_cast<long double>(idx) + 1.0L; // [1, len]
    long double x1_map = remap_index(x1, len, alpha, delta);
    const mp mp_idx = mp(static_cast<double>(x1_map));

    // Use the exact algebraic layout from the Mathematica expression
    const mp t1 = (mp_idx - (mp_len / mp(2)) - mp(1)/mp(2)) * eta;
    const mp tref = (mp(1) - (mp_len / mp(2)) - mp(1)/mp(2)) * eta;
    const mp term1 = (mp(2)/pi) * atan(mp(1) / exp(t1));
    const mp term_ref = (mp(2)/pi) * atan(mp(1) / exp(tref));
    const mp num = (mp(1) - term1) - (mp(1) - term_ref); // equals term_ref - term1

    const mp thalf = ((mp_len / mp(2)) - mp(1)/mp(2)) * eta;
    const mp den = (mp(2)/pi) * atan(exp(thalf)) - (mp(2)/pi) * atan(exp(-thalf));

    mp th = num / den;
    long double out = static_cast<long double>(th);
    if (out < 0.0) out = 0.0;
    if (out > 1.0) out = 1.0;
    return out;
}

// Vectorized version: compute theta for multiple indices efficiently
// by computing Wm1 and other len/Tmax-dependent quantities only once
void theta_of_vec(const std::vector<double>& indices, std::size_t len, double Tmax,
                  std::vector<long double>& theta_values,
                  double alpha, double delta) {
    const mp pi = dmfe::mp_pi();
    const mp mp_len = mp(len);
    const mp mp_Tmax = mp(Tmax);

    // Compute len/Tmax-dependent quantities once
    const mp argW = -mp(1) / (mp(10) * mp_Tmax);
    const mp Wm1 = dmfe::lambertWm1_mp(argW);
    const mp eta = -(mp(2) / mp_len) * Wm1;

    const mp tref = (mp(1) - (mp_len / mp(2)) - mp(1)/mp(2)) * eta;
    const mp term_ref = (mp(2)/pi) * atan(mp(1) / exp(tref));

    const mp thalf = ((mp_len / mp(2)) - mp(1)/mp(2)) * eta;
    const mp den = (mp(2)/pi) * atan(exp(thalf)) - (mp(2)/pi) * atan(exp(-thalf));

    // Reserve output space
    theta_values.resize(indices.size());

    // Compute theta for each index
    for (std::size_t k = 0; k < indices.size(); ++k) {
        long double x1 = static_cast<long double>(indices[k]) + 1.0L;
        long double x1_map = remap_index(x1, len, alpha, delta);
        const mp mp_idx = mp(static_cast<double>(x1_map));  // 1-based mapped index
        
        const mp t1 = (mp_idx - (mp_len / mp(2)) - mp(1)/mp(2)) * eta;
        const mp term1 = (mp(2)/pi) * atan(mp(1) / exp(t1));
        const mp num = (mp(1) - term1) - (mp(1) - term_ref);

        mp th = num / den;
        long double out = static_cast<long double>(th);
        if (out < 0.0) out = 0.0;
        if (out > 1.0) out = 1.0;
        theta_values[k] = out;
    }
}

void generate_theta_grid(std::size_t len, double Tmax, std::vector<long double>& theta,
                         double alpha, double delta) {
    // Build indices vector [0, 1, 2, ..., len-1]
    std::vector<double> indices(len);
    for (std::size_t i = 0; i < len; ++i) {
        indices[i] = static_cast<double>(i);
    }
    
    // Use vectorized version for efficiency
    theta_of_vec(indices, len, Tmax, theta, alpha, delta);
    
    // Ensure strictly non-decreasing due to numerical noise
    for (std::size_t i = 1; i < len; ++i) {
        if (theta[i] < theta[i-1]) theta[i] = theta[i-1];
    }
}

// write_theta_grid moved to grid_io.cpp
