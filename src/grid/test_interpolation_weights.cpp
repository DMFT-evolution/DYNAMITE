#include "grid/theta_grid.hpp"
#include "grid/interpolation_weights.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>

using dmfe::grid::compute_barycentric_weights;
using dmfe::grid::BarycentricStencil;
using dmfe::grid::compute_barycentric_rational_weights;
using dmfe::grid::compute_bspline_weights;
using dmfe::grid::BSplineWeights;

static double max_abs(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}

int main() {
    const std::size_t len = 512;
    const double Tmax = 100000.0;
    const int n = 9; // interpolation order

    // Build theta input grid
    std::vector<double> theta;
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

    // Test monomials up to degree n
    double worst_err = 0.0;
    for (int d = 0; d <= n; ++d) {
        // y_j = theta_j^d
        std::vector<double> y(len);
        for (std::size_t j = 0; j < len; ++j) y[j] = std::pow(theta[j], d);

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
        for (std::size_t j = 0; j < len; ++j) y[j] = std::pow(theta[j], d);
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
        for (std::size_t j = 0; j < len; ++j) y[j] = std::pow(theta[j], d);
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
