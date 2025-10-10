#include "grid/interpolation_weights.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace dmfe {
namespace grid {

namespace {

inline int pick_stencil_start(const std::vector<long double>& x, int n, long double xq) {
    const int N = static_cast<int>(x.size());
    const int m = n + 1;
    int hi = static_cast<int>(std::lower_bound(x.begin(), x.end(), xq) - x.begin());
    int start = std::clamp(hi - m / 2, 0, N - m);
    // Small local search to minimize the max distance to stencil ends
    auto spread = [&](int s) -> long double {
        int e = s + n;
        long double a = x[s];
        long double b = x[e];
        long double left = fabsl((long double)xq - a);
        long double right = fabsl((long double)xq - b);
        return left < right ? right : left;
    };
    long double best = spread(start);
    for (int s = std::max(0, start - 2); s <= std::min(N - m, start + 2); ++s) {
        long double val = spread(s);
        if (val < best) { best = val; start = s; }
    }
    return start;
}

inline void barycentric_node_weights(const std::vector<long double>& nodes,
                                     std::vector<long double>& w) {
    const int m = static_cast<int>(nodes.size());
    w.assign(m, 1.0L);
    for (int j = 0; j < m; ++j) {
        long double denom = 1.0L;
        long double xj = nodes[j];
        for (int k = 0; k < m; ++k) if (k != j) {
            denom *= (xj - nodes[k]);
        }
        w[j] = 1.0L / denom;
    }
}

// Floater–Hormann weights on a local window xn of size m and order d.
// Weights depend only on xn and d (not on the query xq).
inline void floater_hormann_local_weights(const std::vector<long double>& xn,
                                          int d,
                                          std::vector<long double>& wfh) {
    const int m = static_cast<int>(xn.size());
    wfh.assign(m, 0.0L);
    if (m == 0) return;

    for (int j = 0; j < m; ++j) {
        // r is the local offset of x_j within a (d+1)-subset that contains j
        const int rmin = std::max(0, j - (m - d - 1));
        const int rmax = std::min(d, j);
        long double sum = 0.0L;
        for (int r = rmin; r <= rmax; ++r) {
            const int i = j - r; // subset starts at i, runs i..i+d, contains j at offset r
            long double prod = 1.0L;
            bool zero = false;
            for (int k = 0; k <= d; ++k) {
                if (k == r) continue;
                long double diff = xn[j] - xn[i + k];
                if (diff == 0.0L) { zero = true; break; } // duplicate nodes
                prod *= diff;
            }
            if (zero || prod == 0.0L) continue;
            // Sign is (-1)^i = (-1)^(j - r)
            long double sgn = (( (j - r) & 1 ) ? -1.0L : 1.0L);
            sum += sgn / prod;
        }
        wfh[j] = sum;
    }
}

} // namespace

std::vector<BarycentricStencil>
compute_barycentric_weights(const std::vector<long double>& x,
                            const std::vector<long double>& xq,
                            int n) {
    const int N = static_cast<int>(x.size());
    const int m = n + 1;
    if (N < m) return {};

    std::vector<BarycentricStencil> out;
    out.reserve(xq.size());

    for (long double q : xq) {
        int start = pick_stencil_start(x, n, q);
        std::vector<long double> xn; xn.reserve(m);
        for (int j = 0; j < m; ++j) xn.push_back(x[start + j]);

        // If q coincides with a node in the stencil, return delta weights
        int exact_idx = -1;
    for (int j = 0; j < m; ++j) { if (q == xn[j]) { exact_idx = j; break; } }

        std::vector<double> alpha(m, 0.0);
        if (exact_idx >= 0) {
            alpha[exact_idx] = 1.0;
        } else {
            std::vector<long double> wloc;
            barycentric_node_weights(xn, wloc);
            long double den = 0.0L;
            std::vector<long double> tmp(m);
            for (int j = 0; j < m; ++j) {
                long double v = wloc[j] / (q - xn[j]);
                tmp[j] = v;
                den += v;
            }
            long double invden = 1.0L / den;
            for (int j = 0; j < m; ++j) alpha[j] = (double)(tmp[j] * invden);
        }
        out.push_back(BarycentricStencil{ start, std::move(alpha) });
    }

    return out;
}

// Backward-compatible overload
std::vector<BarycentricStencil>
compute_barycentric_weights(const std::vector<double>& x,
                            const std::vector<double>& xq,
                            int n) {
    std::vector<long double> xl(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) xl[i] = (long double)x[i];
    return compute_barycentric_weights(xl, xq, n);
}

std::vector<BarycentricStencil>
compute_barycentric_rational_weights(const std::vector<long double>& x,
                                     const std::vector<long double>& xq,
                                     int d,
                                     int m) {
    // Proper local Floater–Hormann of order d on a centered window of size m.
    const int N = (int)x.size();
    if (d < 0) d = 0;
    if (m < d + 1) m = d + 1;
    if (N < m) return {};

    std::vector<BarycentricStencil> out;
    out.reserve(xq.size());

    std::vector<long double> xn; xn.reserve(m);
    std::vector<long double> wfh; wfh.reserve(m);
    std::vector<long double> tmp; tmp.reserve(m);

    for (long double q : xq) {
        // Choose centered window of size m
        int hi = (int)(std::lower_bound(x.begin(), x.end(), q) - x.begin());
        int start = std::clamp(hi - m / 2, 0, N - m);

        xn.clear(); xn.reserve(m);
        for (int j = 0; j < m; ++j) xn.push_back(x[start + j]);

        // Exact hit (with tolerance relative to local span)
        long double span = std::max(1.0L, xn.back() - xn.front());
        long double tol = 64 * std::numeric_limits<long double>::epsilon() * span;
        int hit = -1;
        for (int j = 0; j < m; ++j) {
            if (fabsl(q - xn[j]) <= tol) { hit = j; break; }
        }

        std::vector<double> alpha(m, 0.0);
        if (hit >= 0) {
            alpha[hit] = 1.0;
            out.push_back(BarycentricStencil{ start, std::move(alpha) });
            continue;
        }

        // Local FH weights on this window
        floater_hormann_local_weights(xn, d, wfh);

        // Barycentric evaluation on the window
        long double den = 0.0L;
        tmp.assign(m, 0.0L);
        for (int j = 0; j < m; ++j) {
            long double v = wfh[j] / (q - xn[j]);
            tmp[j] = v;
            den += v;
        }

        if (den == 0.0L) {
            // Fallback (should be rare): use polynomial barycentric on this window
            std::vector<long double> wloc;
            barycentric_node_weights(xn, wloc);
            long double den2 = 0.0L;
            for (int j = 0; j < m; ++j) { tmp[j] = wloc[j] / (q - xn[j]); den2 += tmp[j]; }
            long double invden2 = 1.0L / den2;
            for (int j = 0; j < m; ++j) alpha[j] = (double)(tmp[j] * invden2);
        } else {
            long double invden = 1.0L / den;
            for (int j = 0; j < m; ++j) alpha[j] = (double)(tmp[j] * invden);
        }

        out.push_back(BarycentricStencil{ start, std::move(alpha) });
    }
    return out;
}

// Backward-compatible overload
std::vector<BarycentricStencil>
compute_barycentric_rational_weights(const std::vector<double>& x,
                                     const std::vector<double>& xq,
                                     int d,
                                     int m) {
    std::vector<long double> xl(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) xl[i] = (long double)x[i];
    return compute_barycentric_rational_weights(xl, xq, d, m);
}

// -------------------- B-spline interpolation (global) --------------------
namespace {

// Build open knot vector with endpoints clamped p+1 times and
// interior knots chosen from the input nodes (first N - p - 1 interior nodes).
static std::vector<long double>
build_open_knot_vector(const std::vector<long double>& x, int p) {
    const int N = static_cast<int>(x.size());
    if (N < p + 1) throw std::invalid_argument("B-spline: need N >= p+1");
    const int mBasis = N; // number of basis functions equals number of data points
    const int K = mBasis + p + 1; // number of knots
    std::vector<long double> t(K);
    // Clamp endpoints to x-front/back (open clamped)
    for (int i = 0; i <= p; ++i) t[i] = x.front();
    for (int i = 0; i <= p; ++i) t[K - 1 - i] = x.back();
    // Interior knots by averaging p consecutive parameter values (use u = x)
    // t[j] = (u_{j-p} + ... + u_{j-1}) / p for j = p+1..n (n = N-1)
    const int n = N - 1;
    for (int j = p + 1; j <= n; ++j) {
        long double sum = 0.0L;
    for (int i = j - p; i <= j - 1; ++i) sum += x[i];
        t[j] = sum / (long double)p;
    }
    return t;
}

// Find span such that t[span] <= u < t[span+1] (The NURBS Book Algorithm A2.1)
static int find_span(int mBasis, int p, long double u, const std::vector<long double>& t) {
    const int n = mBasis - 1;
    if (u >= t[n + 1]) return n; // clamp right
    if (u <= t[p]) return p;     // clamp left
    int low = p;
    int high = n + 1;
    int mid = (low + high) / 2;
    while (!(u >= t[mid] && u < t[mid + 1])) {
        if (u < t[mid]) high = mid; else low = mid;
        mid = (low + high) / 2;
    }
    return mid;
}

// Basis functions N_{i-p..i,p}(u) (Algorithm A2.2). Returns vector size p+1.
static void basis_funs(int span, long double u, int p, const std::vector<long double>& t, std::vector<long double>& N) {
    N.assign(p + 1, 0.0L);
    std::vector<long double> left(p + 1), right(p + 1);
    N[0] = 1.0L;
    for (int j = 1; j <= p; ++j) {
        left[j] = u - t[span + 1 - j];
        right[j] = t[span + j] - u;
        long double saved = 0.0L;
        for (int r = 0; r < j; ++r) {
            long double denom = right[r + 1] + left[j - r];
            long double temp = (denom != 0.0L) ? (N[r] / denom) : 0.0L;
            N[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        N[j] = saved;
    }
}

// Dense LU with partial pivoting in long double for square matrix A (row-major)
static void lu_factor(std::vector<long double>& A, int n, std::vector<int>& piv) {
    piv.resize(n);
    for (int i = 0; i < n; ++i) piv[i] = i;
    for (int k = 0; k < n; ++k) {
        // Pivot
        int p = k;
        long double amax = std::fabs(A[(std::size_t)k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            long double v = std::fabs(A[(std::size_t)i * n + k]);
            if (v > amax) { amax = v; p = i; }
        }
        if (amax == 0.0L) throw std::runtime_error("LU: singular matrix");
        if (p != k) {
            for (int j = 0; j < n; ++j) std::swap(A[(std::size_t)k * n + j], A[(std::size_t)p * n + j]);
            std::swap(piv[k], piv[p]);
        }
        // Eliminate
        for (int i = k + 1; i < n; ++i) {
            A[(std::size_t)i * n + k] /= A[(std::size_t)k * n + k];
            long double lik = A[(std::size_t)i * n + k];
            for (int j = k + 1; j < n; ++j) {
                A[(std::size_t)i * n + j] -= lik * A[(std::size_t)k * n + j];
            }
        }
    }
}

// Solve A x = b using LU factorization (A overwritten by lu_factor), piv from lu_factor
static void lu_solve(const std::vector<long double>& LU, int n, const std::vector<int>& piv,
                     const std::vector<long double>& b, std::vector<long double>& x) {
    // Apply row permutation: x = P * b, where piv encodes final row order
    x.resize(n);
    for (int i = 0; i < n; ++i) x[i] = b[piv[i]];
    // Forward solve L y = Pb
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) x[i] -= LU[(std::size_t)i * n + j] * x[j];
    }
    // Backward solve U x = y
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) x[i] -= LU[(std::size_t)i * n + j] * x[j];
        x[i] /= LU[(std::size_t)i * n + i];
    }
}

} // namespace

struct BSplineWeights;

std::vector<BSplineWeights>
compute_bspline_weights(const std::vector<long double>& x,
                        const std::vector<long double>& xq,
                        int p) {
    const int N = static_cast<int>(x.size());
    if (N < p + 1) throw std::invalid_argument("B-spline: need at least p+1 nodes");
    // Build open knot vector in x-domain
    auto knots = build_open_knot_vector(x, p);
    const int mBasis = N; // number of basis functions equals N

    // Build dense collocation matrix A (N x N) in long double
    std::vector<long double> A((std::size_t)N * N, 0.0L);
    std::vector<long double> Nvals(p + 1);
    for (int i = 0; i < N; ++i) {
    long double u = x[i];
        int span = find_span(mBasis, p, u, knots);
        basis_funs(span, u, p, knots, Nvals);
        int first = span - p;
        for (int j = 0; j <= p; ++j) {
            int col = first + j;
            if (col >= 0 && col < N) A[(std::size_t)i * N + col] = Nvals[j];
        }
    }

    // Factor A^T for weight solves (since w solves A^T w = e)
    // Build AT explicitly to reuse the same LU solver
    std::vector<long double> AT((std::size_t)N * N);
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) AT[(std::size_t)i * N + j] = A[(std::size_t)j * N + i];
    std::vector<int> pivT;
    lu_factor(AT, N, pivT);

    std::vector<BSplineWeights> out;
    out.reserve(xq.size());

    std::vector<long double> rhs(N, 0.0L), sol(N);
    for (long double xqi : xq) {
        // Build basis vector e at xq (nonzeros in a block of size p+1)
        std::fill(rhs.begin(), rhs.end(), 0.0L);
        long double u = xqi;
        int span = find_span(mBasis, p, u, knots);
        basis_funs(span, u, p, knots, Nvals);
        int first = span - p;
        for (int j = 0; j <= p; ++j) {
            int col = first + j;
            if (col >= 0 && col < N) rhs[col] = Nvals[j];
        }
        // Solve AT * w = rhs
        lu_solve(AT, N, pivT, rhs, sol);
        BSplineWeights W;
        W.w.resize(N);
        for (int i = 0; i < N; ++i) W.w[i] = (double)sol[i];
        out.push_back(std::move(W));
    }

    return out;
}

// Backward-compatible overload
std::vector<BSplineWeights>
compute_bspline_weights(const std::vector<double>& x,
                        const std::vector<double>& xq,
                        int n) {
    std::vector<long double> xl(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) xl[i] = (long double)x[i];
    return compute_bspline_weights(xl, xq, n);
}

} // namespace grid
} // namespace dmfe
