#pragma once

#include <vector>
#include <cstddef>

namespace dmfe {
namespace grid {

// Fixed-width local barycentric Lagrange interpolation weights
// for an n-th order polynomial (stencil size = n+1) on a 1D, strictly
// increasing but potentially highly non-uniform input grid.
//
// For each query xq, we choose a contiguous stencil of size n+1 around xq
// and compute the final interpolation weights alpha_j(xq) such that
//     f(xq) ≈ sum_{j=0..n} alpha[j] * f(x[start + j]).
// If xq coincides with a node, alpha is the corresponding Kronecker delta.
//
// Notes:
// - We compute intermediate quantities in long double to improve robustness
//   on irregular grids, then store weights as double.
// - For polynomials of degree ≤ n, the reconstruction is exact in exact
//   arithmetic for any stencil; numerically, errors are at roundoff.

struct BarycentricStencil {
    int start;                 // index of first node in the stencil
    std::vector<double> alpha; // weights for values at x[start + j], j=0..n
};

// Compute weights for all queries in xq using local degree-n barycentric Lagrange.
// Preconditions: x must be strictly increasing and contain at least n+1 points.
std::vector<BarycentricStencil>
compute_barycentric_weights(const std::vector<long double>& x,    // input nodes
                            const std::vector<long double>& xq,   // query points (long double)
                            int n);                               // interpolation order

// Backward-compatible overload
std::vector<BarycentricStencil>
compute_barycentric_weights(const std::vector<double>& x,
                            const std::vector<double>& xq,
                            int n);

// Compute weights for all queries using a local barycentric rational (Floater–Hormann)
// variant with parameter d. For a contiguous stencil size of d+1, this reduces to
// the degree-d polynomial case; this interface is provided so callers can later
// switch to true FH blending with wider windows if desired.
// Floater–Hormann (rational barycentric) with explicit stencil size m (>= d+1).
// For each query, we pick a contiguous window of size m and blend all degree-d
// local Lagrange barycentric formulas within that window. When m==d+1 this
// reduces to the polynomial barycentric case on a single stencil.
std::vector<BarycentricStencil>
compute_barycentric_rational_weights(const std::vector<long double>& x,   // input nodes
                                     const std::vector<long double>& xq,  // query points (long double)
                                     int d,                                // FH order (degree)
                                     int m);                               // stencil size (>= d+1)

// Backward-compatible overload
std::vector<BarycentricStencil>
compute_barycentric_rational_weights(const std::vector<double>& x,
                                     const std::vector<double>& xq,
                                     int d,
                                     int m);

// Backward-compatible overload: m defaults to d+1
inline std::vector<BarycentricStencil>
compute_barycentric_rational_weights(const std::vector<long double>& x,
                                     const std::vector<long double>& xq,
                                     int d) {
    return compute_barycentric_rational_weights(x, xq, d, d + 1);
}

// Convenience overload: accept double xq and promote to long double for internal use
inline std::vector<BarycentricStencil>
compute_barycentric_rational_weights(const std::vector<long double>& x,
                                     const std::vector<double>& xq,
                                     int d) {
    std::vector<long double> xq_ld(xq.begin(), xq.end());
    return compute_barycentric_rational_weights(x, xq_ld, d, d + 1);
}

inline std::vector<BarycentricStencil>
compute_barycentric_rational_weights(const std::vector<double>& x,
                                     const std::vector<double>& xq,
                                     int d) {
    return compute_barycentric_rational_weights(x, xq, d, d + 1);
}

// Global n-th order B-spline interpolation weights.
// Builds a clamped knot vector from the input nodes (parameterized by x directly),
// forms the interpolation matrix A_ij = N_j(x_i), factors it once, then for each
// query computes the weight vector w such that s(xq) = sum_j w[j] * y_j for any data y.
// Note: weights are generally dense (global), not local like polynomial barycentric.
struct BSplineWeights {
    std::vector<double> w; // length x.size()
};

std::vector<BSplineWeights>
compute_bspline_weights(const std::vector<long double>& x,   // input nodes (strictly increasing)
                        const std::vector<long double>& xq,  // query points (long double)
                        int n);                               // spline degree

// Local uniform index-coordinate B-spline *kernel* weights.
//
// This is intended for highly non-uniform x grids where you already have an accurate
// inverse map x -> u (a fractional 1-based index in [1,N]) for each query.
//
// Given samples y[i] defined on a uniform index coordinate u=i+1 (1-based), this
// routine produces compact (degree+1)-point stencils to evaluate a uniform B-spline
// kernel at fractional u.
//
// Notes:
// - This is a local, compact-support spline kernel on the *index* coordinate.
// - It does not solve the global B-spline interpolation system; i.e. it behaves like
//   a stable smoothing/interpolation kernel using the samples directly as coefficients.
// - Output format matches other local methods: contiguous indices starting at `start`.
// - Inputs u_q must be 1-based fractional indices (like posA1y/posA2y/posB2y).
std::vector<BarycentricStencil>
compute_index_bspline_weights(const std::vector<double>& u_q, // 1-based fractional indices in [1,N]
                              std::size_t N,                  // number of samples (theta grid length)
                              int degree);                    // B-spline degree (>=0)

// Local uniform index-coordinate polynomial interpolation weights (Lagrange), using
// barycentric evaluation on a contiguous stencil of size n+1.
//
// This is a *true interpolant* in index space: for integer u=i (1-based), it reproduces
// the sample exactly (up to floating point tolerance) when that point is within the chosen stencil.
//
// Inputs u_q must be 1-based fractional indices in [1,N] (like posA1y/posA2y/posB2y).
// Output format matches other local methods: contiguous indices starting at `start`.
std::vector<BarycentricStencil>
compute_index_poly_weights(const std::vector<double>& u_q, // 1-based fractional indices in [1,N]
                           std::size_t N,                  // number of samples
                           int n);                         // polynomial degree/order (>=0)

// Local uniform index-coordinate barycentric rational interpolation weights (Floater–Hormann).
//
// Inputs u_q are 1-based fractional indices in [1,N]. Internally we work on the uniform
// 0-based coordinate t=u-1 with implicit nodes 0,1,...,N-1.
//
// Parameters:
// - d: FH order (degree parameter), must satisfy d>=0.
// - m: window size (number of points used), must satisfy m>=d+1.
//
// Output format matches other local methods: contiguous indices starting at `start` and
// weights alpha of length m.
std::vector<BarycentricStencil>
compute_index_rational_weights(const std::vector<double>& u_q, // 1-based fractional indices in [1,N]
                               std::size_t N,                  // number of samples
                               int d,                          // FH order
                               int m);                         // window size (>= d+1)

// Local uniform index-coordinate Hermite interpolation weights (piecewise cubic).
//
// This is a *true interpolant* in index space: it interpolates the samples at integer
// indices (up to floating point tolerance). Between nodes, it uses a cubic Hermite
// segment on each interval [i,i+1].
//
// The "order" parameter controls the size of the local stencil (m = order+1) used
// to estimate endpoint slopes via polynomial interpolation derivatives on that stencil.
// Larger order typically improves accuracy while keeping the method stable and local.
//
// Notes:
// - Output format matches other local methods: contiguous indices starting at `start`.
// - Inputs u_q must be 1-based fractional indices in [1,N] (like posA1y/posA2y/posB2y).
// - For very small N, we fall back to index-poly.
std::vector<BarycentricStencil>
compute_index_hermite_weights(const std::vector<double>& u_q, // 1-based fractional indices in [1,N]
                              std::size_t N,                  // number of samples
                              int order);                     // interpolation order n (stencil size n+1)

// Backward-compatible convenience overload: m defaults to d+1.
inline std::vector<BarycentricStencil>
compute_index_rational_weights(const std::vector<double>& u_q,
                               std::size_t N,
                               int d) {
    return compute_index_rational_weights(u_q, N, d, d + 1);
}

// Backward-compatible overloads for double xq
inline std::vector<BarycentricStencil>
compute_barycentric_weights(const std::vector<long double>& x,
                            const std::vector<double>& xq,
                            int n) {
    std::vector<long double> xq_ld(xq.begin(), xq.end());
    return compute_barycentric_weights(x, xq_ld, n);
}

inline std::vector<BarycentricStencil>
compute_barycentric_rational_weights(const std::vector<long double>& x,
                                     const std::vector<double>& xq,
                                     int d,
                                     int m) {
    std::vector<long double> xq_ld(xq.begin(), xq.end());
    return compute_barycentric_rational_weights(x, xq_ld, d, m);
}

inline std::vector<BSplineWeights>
compute_bspline_weights(const std::vector<long double>& x,
                        const std::vector<double>& xq,
                        int n) {
    std::vector<long double> xq_ld(xq.begin(), xq.end());
    return compute_bspline_weights(x, xq_ld, n);
}

// Backward-compatible overload
std::vector<BSplineWeights>
compute_bspline_weights(const std::vector<double>& x,
                        const std::vector<double>& xq,
                        int n);

} // namespace grid
} // namespace dmfe
