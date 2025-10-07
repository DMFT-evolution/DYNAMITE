#include "grid/integration.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "grid/grid_io.hpp"
// High-precision arithmetic for stable small linear solves
#include <boost/multiprecision/cpp_dec_float.hpp>

namespace {
// Solve small linear system A x = b via Gaussian elimination with partial pivoting.
// A is (m x m), stored row-major, b and x are size m.
template <typename T>
static void solve_small_t(std::vector<T>& A, std::vector<T>& b, int m) {
    using boost::multiprecision::abs; // enable ADL for multiprecision types
    using std::abs;
    for (int k = 0; k < m; ++k) {
        // pivot
        int piv = k;
        T amax = abs(A[k * m + k]);
        for (int i = k + 1; i < m; ++i) {
            T v = abs(A[i * m + k]);
            if (v > amax) { amax = v; piv = i; }
        }
        if (amax == T(0)) continue; // singular; leave as is
        if (piv != k) {
            for (int j = k; j < m; ++j) std::swap(A[k * m + j], A[piv * m + j]);
            std::swap(b[k], b[piv]);
        }
        T diag = A[k * m + k];
        for (int j = k; j < m; ++j) A[k * m + j] /= diag;
        b[k] /= diag;
        for (int i = 0; i < m; ++i) {
            if (i == k) continue;
            T f = A[i * m + k];
            if (f == T(0)) continue;
            for (int j = k; j < m; ++j) A[i * m + j] -= f * A[k * m + j];
            b[i] -= f * b[k];
        }
    }
}

} // namespace

void compute_integration_weights(const std::vector<double>& theta,
                                 int order,
                                 std::vector<double>& w) {
    using mp_t = boost::multiprecision::cpp_dec_float_100;
    const std::size_t N = theta.size();
    if (N == 0) { w.clear(); return; }
    if (order < 1) order = 1;
    if (order > 8) order = 8;

    w.assign(N, 0.0);
    // use extended precision internally
    std::vector<long double> wl(N, 0.0L);
    std::vector<long double> comp(N, 0.0L); // Kahan compensation per node

    // Composite local polynomial rule: for each interval [t_i, t_{i+1}],
    // build weights a_j such that sum_j a_j p(t_j) = ∫_{t_i}^{t_{i+1}} p(t) dt for all polynomials p of degree <= order.
    // Use a stencil S of size M = min(order+1, N) nodes near the interval; center symmetrically around [i,i+1] when possible.
    const int M = std::min<int>(order + 1, (int)N);
    // For stability, limit at ends by sliding window.
    for (std::size_t i = 0; i + 1 < N; ++i) {
        double a = theta[i], b = theta[i + 1];
        double h = b - a;
        if (h <= 0.0) continue;
        // symmetric stencil in index space around the interval [i, i+1]
        int j0 = static_cast<int>(i) - (M/2 - 1);
        if (j0 < 0) j0 = 0;
        if (j0 + M > (int)N) j0 = (int)N - M;

        // Center-and-scale absolute moment formulation: y = (t - c) / s
        const mp_t a_mp = mp_t(a);
        const mp_t b_mp = mp_t(b);
        const mp_t c = (a_mp + b_mp) / mp_t(2);
        // scale s as the local stencil span for robustness, but not smaller than h
        const double span_d = theta[j0 + M - 1] - theta[j0];
        const mp_t s = mp_t(std::max(h, span_d));
        const mp_t ya = (a_mp - c) / s;
        const mp_t yb = (b_mp - c) / s;

        std::vector<mp_t> A(M * M);
        std::vector<mp_t> bvec(M);
        std::vector<mp_t> yk(M);
        for (int k2 = 0; k2 < M; ++k2) {
            yk[k2] = (mp_t(theta[j0 + k2]) - c) / s;
        }
        // Build Vandermonde in powers of y, and exact moments over [ya, yb]
        for (int r = 0; r < M; ++r) {
            // moment of y^r over [ya, yb] in t-units: ∫ (s y + c)' dt = s ∫ y^r dy = s * (yb^{r+1} - ya^{r+1})/(r+1)
            mp_t ypow_a = mp_t(1), ypow_b = mp_t(1);
            for (int k = 0; k < r + 1; ++k) { ypow_a *= ya; ypow_b *= yb; }
            bvec[r] = s * (ypow_b - ypow_a) / mp_t(r + 1);
            for (int k2 = 0; k2 < M; ++k2) {
                if (r == 0) {
                    A[r * M + k2] = mp_t(1);
                } else {
                    mp_t p = mp_t(1);
                    for (int tpow = 0; tpow < r; ++tpow) p *= yk[k2];
                    A[r * M + k2] = p;
                }
            }
        }
        solve_small_t(A, bvec, M);
        // Accumulate directly (no extra h factor; bvec already in t-units)
        for (int k2 = 0; k2 < M; ++k2) {
            long double increment = static_cast<long double>(bvec[k2]);
            long double y = increment - comp[j0 + k2];
            long double t = wl[j0 + k2] + y;
            comp[j0 + k2] = (t - wl[j0 + k2]) - y;
            wl[j0 + k2] = t;
        }
    }

    // Normalize to ensure exactness for constants: sum w = 1 when theta[0]=0, theta[-1]=1
    long double sum = 0.0L, csum = 0.0L;
    for (long double v : wl) {
        long double y = v - csum;
        long double t = sum + y;
        csum = (t - sum) - y;
        sum = t;
    }
    if (sum != 0.0L) {
        for (auto& v : wl) v /= sum;
    }
    // Enforce non-negative small eps at ends
    if (!wl.empty()) {
        if (wl.front() < 0) wl.front() = 0.0L;
        if (wl.back() < 0) wl.back() = 0.0L;
    }

    // cast back to double
    for (std::size_t i2 = 0; i2 < N; ++i2) w[i2] = static_cast<double>(wl[i2]);
}

// write_integration_weights moved to grid_io.cpp; reading centralized in grid_io.cpp

bool validate_integration_weights(const std::vector<double>& weights,
                                  std::size_t len,
                                  const std::string& subdir,
                                  double tol,
                                  double& maxAbsDiff,
                                  std::size_t& mismatches) {
    maxAbsDiff = 0.0; mismatches = 0;
    std::string path = std::string("Grid_data/") + subdir + "/int.dat";
    std::vector<double> ref;
    if (!read_vector_file(path, ref)) return false;
    if (ref.size() != len) return false;
    auto update = [&](double a, double b){
        double d = std::abs(a - b);
        if (d > maxAbsDiff) maxAbsDiff = d;
        if (d > tol) ++mismatches;
    };
    std::size_t n = std::min<std::size_t>(len, weights.size());
    for (std::size_t i = 0; i < n; ++i) update(weights[i], ref[i]);
    return mismatches == 0;
}
