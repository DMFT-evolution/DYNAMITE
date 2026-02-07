#include "grid/theta_grid.hpp"
#include "grid/interpolation_weights.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>

static long double pow_int(long double x, int n) {
    long double r = 1.0L;
    for (int i = 0; i < n; ++i) r *= x;
    return r;
}

using dmfe::grid::compute_barycentric_weights;
using dmfe::grid::BarycentricStencil;
using dmfe::grid::compute_barycentric_rational_weights;
using dmfe::grid::compute_bspline_weights;
using dmfe::grid::BSplineWeights;
using dmfe::grid::compute_index_bspline_weights;
using dmfe::grid::compute_index_poly_weights;
using dmfe::grid::compute_index_rational_weights;
using dmfe::grid::compute_index_hermite_weights;

int main() {
    const std::size_t len = 512;
    const double Tmax = 100000.0;
    const int n = 9; // interpolation order

    // Build theta input grid
    std::vector<long double> theta;
    theta.reserve(len);
    generate_theta_grid(len, Tmax, theta);

    // Build output grid: 10 equidistant points in [0,1]
    std::vector<double> xq(10);
    for (int i = 0; i < 10; ++i) xq[i] = i / 9.0; // includes 0 and 1

    // Precompute weights (polynomial, rational, and B-spline)
    auto stencils_poly = compute_barycentric_weights(theta, xq, n);
    auto stencils_rat  = compute_barycentric_rational_weights(theta, xq, n);
    // Also test Floater–Hormann with larger stencil size m > n+1
    const int mFH = n + 5;
    auto stencils_rat_wide = compute_barycentric_rational_weights(theta, xq, n, mFH);
    auto weights_bs    = compute_bspline_weights(theta, xq, n);

    // Index-space B-spline kernel weights sanity checks
    {
        const int p = 5;
        std::vector<double> uq = {1.0, 1.25, 2.5, static_cast<double>(len) - 0.1, static_cast<double>(len)};
        auto st_idx = compute_index_bspline_weights(uq, len, p);
        if (st_idx.size() != uq.size()) {
            std::cerr << "index-bspline FAILED: wrong output size\n";
            return 1;
        }
        const int m = p + 1;
        std::vector<double> y(len, 3.141592653589793);
        for (std::size_t qi = 0; qi < st_idx.size(); ++qi) {
            const auto& st = st_idx[qi];
            if ((int)st.alpha.size() != m) {
                std::cerr << "index-bspline FAILED: wrong alpha size\n";
                return 1;
            }
            if (st.start < 0 || st.start > (int)len - m) {
                std::cerr << "index-bspline FAILED: start out of range\n";
                return 1;
            }
            long double sum = 0.0L;
            long double acc = 0.0L;
            for (int j = 0; j < m; ++j) {
                const double a = st.alpha[j];
                if (a < -1e-14) {
                    std::cerr << "index-bspline FAILED: negative weight\n";
                    return 1;
                }
                sum += (long double)a;
                acc += (long double)a * (long double)y[std::size_t(st.start + j)];
            }
            if (std::abs((double)sum - 1.0) > 5e-13) {
                std::cerr << "index-bspline FAILED: weights do not sum to 1\n";
                return 1;
            }
            if (std::abs((double)acc - y[0]) > 5e-13) {
                std::cerr << "index-bspline FAILED: constant not reproduced\n";
                return 1;
            }
        }
        std::cout << "[idx ] Basic sanity checks PASSED\n";
    }

    // Index-space B-spline: degree-1 should reduce to linear interpolation in u
    {
        const int p = 1;
        std::vector<double> uq = {1.0, 1.2, 2.5, 17.75, 101.125, static_cast<double>(len) - 0.6, static_cast<double>(len)};
        auto st = compute_index_bspline_weights(uq, len, p);
        if (st.size() != uq.size()) {
            std::cerr << "index-bspline(p=1) FAILED: wrong output size\n";
            return 1;
        }

        const long double a = 0.123456789L;
        const long double b = -2.25L;
        std::vector<double> y(len);
        for (std::size_t i = 0; i < len; ++i) {
            const long double u = static_cast<long double>(i) + 1.0L;
            y[i] = (double)(a * u + b);
        }

        const double tol = 5e-13;
        for (std::size_t qi = 0; qi < uq.size(); ++qi) {
            const auto& S = st[qi];
            if (S.alpha.size() != 2) {
                std::cerr << "index-bspline(p=1) FAILED: wrong alpha size\n";
                return 1;
            }
            long double acc = 0.0L;
            for (int j = 0; j < 2; ++j) {
                acc += (long double)S.alpha[j] * (long double)y[std::size_t(S.start + j)];
            }
            const long double exact = a * (long double)uq[qi] + b;
            const long double err = fabsl(acc - exact);
            const long double bound = (long double)tol * (fabsl(exact) > 1.0L ? fabsl(exact) : 1.0L);
            if (err > bound) {
                std::cerr << "index-bspline(p=1) FAILED: err=" << (double)err << " > bound=" << (double)bound << "\n";
                return 1;
            }
        }

        std::cout << "[idx1] Degree-1 exactness PASSED\n";
    }

    // Index-space polynomial (Lagrange) weights: exactness on polynomials in u
    {
        const int p = 9; // degree
        std::vector<double> uq = {1.0, 1.25, 2.5, 7.75, 13.125, static_cast<double>(len) - 0.1, static_cast<double>(len)};
        auto st = compute_index_poly_weights(uq, len, p);
        if (st.size() != uq.size()) {
            std::cerr << "index-poly FAILED: wrong output size\n";
            return 1;
        }
        const int m = p + 1;
        const double tol = 5e-13;
        const long double invNm1 = (len > 1) ? (1.0L / (static_cast<long double>(len) - 1.0L)) : 1.0L;
        for (int d = 0; d <= p; ++d) {
            std::vector<double> y(len);
            for (std::size_t i = 0; i < len; ++i) {
                // Use a normalized coordinate z in [0,1] to avoid huge magnitudes.
                const long double u = static_cast<long double>(i) + 1.0L;
                const long double z = (u - 1.0L) * invNm1;
                y[i] = (double)pow_int(z, d);
            }
            for (std::size_t qi = 0; qi < uq.size(); ++qi) {
                const auto& S = st[qi];
                if ((int)S.alpha.size() != m) {
                    std::cerr << "index-poly FAILED: wrong alpha size\n";
                    return 1;
                }
                long double acc = 0.0L;
                for (int j = 0; j < m; ++j) {
                    acc += (long double)S.alpha[j] * (long double)y[std::size_t(S.start + j)];
                }
                const long double zq = (static_cast<long double>(uq[qi]) - 1.0L) * invNm1;
                const long double exact = pow_int(zq, d);
                const long double err = fabsl(acc - exact);
                const long double bound = static_cast<long double>(tol) * (fabsl(exact) > 1.0L ? fabsl(exact) : 1.0L);
                if (err > bound) {
                    std::cerr << "index-poly FAILED: d=" << d << " err=" << (double)err << " > bound=" << (double)bound << "\n";
                    return 1;
                }
            }
        }
        std::cout << "[ipol] Polynomial exactness PASSED\n";
    }

    // Index-space rational (Floater–Hormann): for m=d+1 it should reduce to the polynomial interpolant
    {
        const int d = 9;
        const int m = d + 1;
        std::vector<double> uq = {1.0, 1.25, 2.5, 7.75, 13.125, static_cast<double>(len) - 0.1, static_cast<double>(len)};
        auto st_poly = compute_index_poly_weights(uq, len, d);
        auto st_rat = compute_index_rational_weights(uq, len, d, m);
        if (st_rat.size() != st_poly.size()) {
            std::cerr << "index-rational FAILED: size mismatch\n";
            return 1;
        }
        const double tolW = 5e-13;
        for (std::size_t qi = 0; qi < uq.size(); ++qi) {
            if (st_rat[qi].start != st_poly[qi].start) {
                std::cerr << "index-rational FAILED: start mismatch\n";
                return 1;
            }
            if (st_rat[qi].alpha.size() != st_poly[qi].alpha.size()) {
                std::cerr << "index-rational FAILED: alpha size mismatch\n";
                return 1;
            }
            for (std::size_t j = 0; j < st_rat[qi].alpha.size(); ++j) {
                const double err = std::abs(st_rat[qi].alpha[j] - st_poly[qi].alpha[j]);
                if (err > tolW) {
                    std::cerr << "index-rational FAILED: weight mismatch err=" << err << "\n";
                    return 1;
                }
            }
        }
        std::cout << "[irat] Reduction-to-poly PASSED\n";
    }

    // Index-space Hermite: exact for constants and linears in u for any stencil size
    {
        std::vector<double> uq = {1.0, 1.25, 2.5, 7.75, 13.125, static_cast<double>(len) - 0.1, static_cast<double>(len)};
        const int orders[] = {3, 9};
        for (int p : orders) {
            auto st = compute_index_hermite_weights(uq, len, p);
            if (st.size() != uq.size()) {
                std::cerr << "index-hermite FAILED: wrong output size\n";
                return 1;
            }
            const int m = p + 1;
            for (int deg = 0; deg <= 1; ++deg) {
                const long double a = (deg == 0) ? 0.0L : 0.25L;
                const long double b = 3.0L;
                std::vector<double> y(len);
                for (std::size_t i = 0; i < len; ++i) {
                    const long double u = static_cast<long double>(i) + 1.0L;
                    y[i] = (double)(a * u + b);
                }
                for (std::size_t qi = 0; qi < uq.size(); ++qi) {
                    const auto& S = st[qi];
                    if ((int)S.alpha.size() != m) {
                        std::cerr << "index-hermite FAILED: wrong alpha size\n";
                        return 1;
                    }
                    long double sumw = 0.0L;
                    long double acc = 0.0L;
                    for (int j = 0; j < m; ++j) {
                        sumw += (long double)S.alpha[j];
                        acc += (long double)S.alpha[j] * (long double)y[std::size_t(S.start + j)];
                    }
                    if (std::abs((double)sumw - 1.0) > 5e-13) {
                        std::cerr << "index-hermite FAILED: weights do not sum to 1\n";
                        return 1;
                    }
                    const long double exact = a * (long double)uq[qi] + b;
                    const long double err = fabsl(acc - exact);
                    if (err > 5e-13L * (fabsl(exact) > 1.0L ? fabsl(exact) : 1.0L)) {
                        std::cerr << "index-hermite FAILED: linear exactness err=" << (double)err << "\n";
                        return 1;
                    }
                }
            }
        }
        std::cout << "[iher] Constant/linear exactness PASSED\n";
    }

    // Test monomials up to degree n
    double worst_err = 0.0;
    for (int d = 0; d <= n; ++d) {
        // y_j = theta_j^d
        std::vector<double> y(len);
    for (std::size_t j = 0; j < len; ++j) y[j] = std::pow(static_cast<double>(theta[j]), d);

        // Evaluate at queries
    for (int qi = 0; qi < (int)xq.size(); ++qi) {
        const BarycentricStencil& st = stencils_poly[qi];
            long double acc = 0.0L;
            for (int k = 0; k < (int)st.alpha.size(); ++k) {
                acc += (long double)st.alpha[k] * (long double)y[st.start + k];
            }
            double approx = (double)acc;
            double exact = std::pow(xq[qi], d);
            double err = std::abs(approx - exact);
            worst_err = std::max(worst_err, err);
        }
    }

    std::cout.setf(std::ios::scientific);
    std::cout.precision(3);
    std::cout << "[poly] Max abs error over d<=" << n << ": " << worst_err << "\n";

    // Expect error near double precision roundoff (~1e-14 to 1e-15)
    const double tol = 5e-13; // conservative bound considering conditioning
    if (worst_err > tol) {
        std::cerr << "Poly FAILED: error " << worst_err << " > tol " << tol << "\n";
        return 1;
    }

    // Repeat check for rational variant (should be identical here)
    worst_err = 0.0;
    for (int d = 0; d <= n; ++d) {
        std::vector<double> y(len);
    for (std::size_t j = 0; j < len; ++j) y[j] = std::pow(static_cast<double>(theta[j]), d);
        for (int qi = 0; qi < (int)xq.size(); ++qi) {
            const BarycentricStencil& st = stencils_rat[qi];
            long double acc = 0.0L;
            for (int k = 0; k < (int)st.alpha.size(); ++k) {
                acc += (long double)st.alpha[k] * (long double)y[st.start + k];
            }
            double approx = (double)acc;
            double exact = std::pow(xq[qi], d);
            double err = std::abs(approx - exact);
            worst_err = std::max(worst_err, err);
        }
    }
    std::cout << "[rat ] Max abs error over d<=" << n << ": " << worst_err << "\n";
    if (worst_err > tol) {
        std::cerr << "Rational FAILED: error " << worst_err << " > tol " << tol << "\n";
        return 1;
    }
    // Wide FH stencil test
    worst_err = 0.0;
    for (int d = 0; d <= n; ++d) {
        std::vector<double> y(len);
    for (std::size_t j = 0; j < len; ++j) y[j] = std::pow(static_cast<double>(theta[j]), d);
        for (int qi = 0; qi < (int)xq.size(); ++qi) {
            const BarycentricStencil& st = stencils_rat_wide[qi];
            long double acc = 0.0L;
            for (int k = 0; k < (int)st.alpha.size(); ++k) {
                acc += (long double)st.alpha[k] * (long double)y[st.start + k];
            }
            double approx = (double)acc;
            double exact = std::pow(xq[qi], d);
            double err = std::abs(approx - exact);
            worst_err = std::max(worst_err, err);
        }
    }
    std::cout << "[rat+m] Max abs error over d<=" << n << ": " << worst_err << "\n";
    if (worst_err > tol) {
        std::cerr << "Rational (wide) FAILED: error " << worst_err << " > tol " << tol << "\n";
        return 1;
    }
    // B-spline global weights test
    worst_err = 0.0;
    for (int d = 0; d <= n; ++d) {
        std::vector<double> y(len);
        for (std::size_t j = 0; j < len; ++j) y[j] = std::pow(theta[j], d);
        for (int qi = 0; qi < (int)xq.size(); ++qi) {
            const BSplineWeights& W = weights_bs[qi];
            long double acc = 0.0L;
            for (std::size_t k = 0; k < y.size(); ++k) acc += (long double)W.w[k] * (long double)y[k];
            double approx = (double)acc;
            double exact = std::pow(xq[qi], d);
            double err = std::abs(approx - exact);
            worst_err = std::max(worst_err, err);
        }
    }
    std::cout << "[bspl] Max abs error over d<=" << n << ": " << worst_err << "\n";
    if (worst_err > tol) {
        std::cerr << "B-spline FAILED: error " << worst_err << " > tol " << tol << "\n";
        return 1;
    }

    std::cout << "All variants PASSED" << std::endl;
    return 0;
}
