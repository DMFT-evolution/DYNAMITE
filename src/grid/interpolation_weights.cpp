#include "grid/interpolation_weights.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace dmfe {
namespace grid {

namespace {

inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline int reflect_index(int i, int N) {
    if (N <= 1) return 0;
    // Mirror extension: ..., 2,1,0,1,2,3,...,N-2,N-1,N-2,...
    while (i < 0 || i >= N) {
        if (i < 0) {
            i = -i;
        } else {
            i = 2 * (N - 1) - i;
        }
    }
    return i;
}

inline int pick_stencil_start(const std::vector<long double>& x, int n, long double xq) {
    const int N = static_cast<int>(x.size());
    const int m = n + 1;
    int hi = static_cast<int>(std::lower_bound(x.begin(), x.end(), xq) - x.begin());
    int start = hi - m / 2;
    if (start < 0) start = 0;
    if (start > N - m) start = N - m;
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

// Pick a size-m window around xq that minimizes the max distance to the window ends (like poly)
inline int pick_window_start(const std::vector<long double>& x, int m, long double xq) {
    const int N = static_cast<int>(x.size());
    int hi = static_cast<int>(std::lower_bound(x.begin(), x.end(), xq) - x.begin());
    int start = hi - m / 2;
    if (start < 0) start = 0;
    if (start > N - m) start = N - m;
    auto spread = [&](int s) -> long double {
        long double a = x[s];
        long double b = x[s + m - 1];
        long double left  = fabsl(xq - a);
        long double right = fabsl(xq - b);
        return left < right ? right : left;
    };
    long double best = spread(start);
    for (int s = std::max(0, start - 2); s <= std::min(N - m, start + 2); ++s) {
        long double val = spread(s);
        if (val < best) { best = val; start = s; }
    }
    return start;
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
compute_index_bspline_weights(const std::vector<double>& u_q, std::size_t N_in, int degree_in) {
    const int N = static_cast<int>(N_in);
    if (N <= 0) return {};

    int p = degree_in;
    if (p < 0) p = 0;
    if (p > N - 1) p = N - 1;
    const int m = p + 1;
    if (m <= 0) return {};

    // Center the uniform B-spline kernel on integer sample locations.
    // With uniform knots U[i]=i, the degree-p basis N_{i,p} peaks at i + (p+1)/2.
    // Using u = t + shift with shift=(p+1)/2 makes the peak align with t at integers.
    const long double shift = (p >= 1) ? (0.5L * (static_cast<long double>(p) + 1.0L)) : 0.0L;

    std::vector<BarycentricStencil> out;
    out.reserve(u_q.size());

    // Uniform knot sequence U[i] = i (implicit). Evaluate basis functions N_{i,p}(t)
    // for i in [span-p, span], where span = floor(t) (for t in [span, span+1)).
    // We then fold any out-of-range i into the valid domain by accumulating into
    // a contiguous stencil start..start+p with start clamped.
    std::vector<long double> left(p + 1);
    std::vector<long double> right(p + 1);
    std::vector<long double> Nbasis(p + 1);

    const long double tmin = 0.0L;
    const long double tmax = static_cast<long double>(N - 1);

    for (double uq1 : u_q) {
        // Convert 1-based fractional index in [1,N] to 0-based coordinate t in [0, N-1].
        long double t = static_cast<long double>(uq1) - 1.0L;
        if (t <= tmin) {
            std::vector<double> alpha(m, 0.0);
            alpha[0] = 1.0;
            out.push_back(BarycentricStencil{0, std::move(alpha)});
            continue;
        }
        if (t >= tmax) {
            const int start = std::max(0, N - m);
            std::vector<double> alpha(m, 0.0);
            alpha[m - 1] = 1.0;
            out.push_back(BarycentricStencil{start, std::move(alpha)});
            continue;
        }

        const long double u = t + shift;
        int span = static_cast<int>(std::floor(u));

        // Basis function evaluation (Piegl & Tiller basisFuns) with uniform knots U[i]=i.
        Nbasis.assign(p + 1, 0.0L);
        Nbasis[0] = 1.0L;
        for (int j = 1; j <= p; ++j) {
            left[j]  = u - static_cast<long double>(span + 1 - j);
            right[j] = static_cast<long double>(span + j) - u;
            long double saved = 0.0L;
            for (int r = 0; r < j; ++r) {
                const long double denom = right[r + 1] + left[j - r];
                const long double temp = (denom != 0.0L) ? (Nbasis[r] / denom) : 0.0L;
                Nbasis[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            Nbasis[j] = saved;
        }

        const int i0 = span - p; // basis indices are i0..i0+p
        const int start = clamp_int(i0, 0, std::max(0, N - m));
        std::vector<double> alpha(m, 0.0);

        long double sum = 0.0L;
        for (int j = 0; j <= p; ++j) {
            const int i = i0 + j;
            const int ic = reflect_index(i, N);
            const int local = ic - start;
            if (local >= 0 && local < m) {
                alpha[local] += static_cast<double>(Nbasis[j]);
                sum += Nbasis[j];
            }
        }

        // Renormalize to partition of unity after boundary folding.
        if (sum != 0.0L) {
            const double invsum = 1.0 / static_cast<double>(sum);
            for (int j = 0; j < m; ++j) alpha[j] *= invsum;
        }

        out.push_back(BarycentricStencil{start, std::move(alpha)});
    }

    return out;
}

namespace {

inline int pick_uniform_stencil_start(int N, int n, long double t) {
    const int m = n + 1;
    if (N <= m) return 0;

    // Mimic lower_bound on uniform nodes i=0..N-1
    // hi = first i such that i >= t => ceil(t)
    int hi = static_cast<int>(std::ceil(t));
    if (hi < 0) hi = 0;
    if (hi > N - 1) hi = N - 1;

    int start = hi - m / 2;
    if (start < 0) start = 0;
    if (start > N - m) start = N - m;

    auto spread = [&](int s) -> long double {
        const long double a = static_cast<long double>(s);
        const long double b = static_cast<long double>(s + n);
        const long double left = fabsl(t - a);
        const long double right = fabsl(t - b);
        return left < right ? right : left;
    };

    long double best = spread(start);
    for (int s = std::max(0, start - 2); s <= std::min(N - m, start + 2); ++s) {
        long double val = spread(s);
        if (val < best) {
            best = val;
            start = s;
        }
    }
    return start;
}

inline int pick_uniform_window_start(int N, int m, long double t) {
    if (m <= 0) return 0;
    if (N <= m) return 0;

    int hi = static_cast<int>(std::ceil(t));
    if (hi < 0) hi = 0;
    if (hi > N - 1) hi = N - 1;

    int start = hi - m / 2;
    if (start < 0) start = 0;
    if (start > N - m) start = N - m;

    auto spread = [&](int s) -> long double {
        const long double a = static_cast<long double>(s);
        const long double b = static_cast<long double>(s + m - 1);
        const long double left = fabsl(t - a);
        const long double right = fabsl(t - b);
        return left < right ? right : left;
    };

    long double best = spread(start);
    for (int s = std::max(0, start - 2); s <= std::min(N - m, start + 2); ++s) {
        long double val = spread(s);
        if (val < best) {
            best = val;
            start = s;
        }
    }
    return start;
}

} // namespace

std::vector<BarycentricStencil>
compute_index_poly_weights(const std::vector<double>& u_q, std::size_t N_in, int n_in) {
    const int N = static_cast<int>(N_in);
    if (N <= 0) return {};

    int n = n_in;
    if (n < 0) n = 0;
    if (n > N - 1) n = N - 1;
    const int m = n + 1;
    if (m <= 0) return {};

    // Precompute barycentric node weights for nodes 0..n (shift-invariant for contiguous integers)
    std::vector<long double> wloc(m, 1.0L);
    for (int j = 0; j < m; ++j) {
        long double denom = 1.0L;
        const long double xj = static_cast<long double>(j);
        for (int k = 0; k < m; ++k) {
            if (k == j) continue;
            denom *= (xj - static_cast<long double>(k));
        }
        wloc[j] = 1.0L / denom;
    }

    std::vector<BarycentricStencil> out;
    out.reserve(u_q.size());

    const long double tmin = 0.0L;
    const long double tmax = static_cast<long double>(N - 1);
    const long double span = static_cast<long double>(n > 0 ? n : 1);
    const long double tol_hit = 64.0L * std::numeric_limits<long double>::epsilon() * span;

    std::vector<long double> tmp(m, 0.0L);
    std::vector<long double> alpha_ld(m, 0.0L);

    for (double uq1 : u_q) {
        // 1-based fractional index u in [1,N] -> 0-based coordinate t in [0,N-1]
        long double t = static_cast<long double>(uq1) - 1.0L;
        if (t <= tmin) {
            std::vector<double> alpha(m, 0.0);
            alpha[0] = 1.0;
            out.push_back(BarycentricStencil{0, std::move(alpha)});
            continue;
        }
        if (t >= tmax) {
            const int start = std::max(0, N - m);
            std::vector<double> alpha(m, 0.0);
            alpha[m - 1] = 1.0;
            out.push_back(BarycentricStencil{start, std::move(alpha)});
            continue;
        }

        int start = pick_uniform_stencil_start(N, n, t);

        // Exact hit within tolerance: return Kronecker delta in this stencil if possible
        int hit = -1;
        for (int j = 0; j < m; ++j) {
            long double node = static_cast<long double>(start + j);
            if (fabsl(t - node) <= tol_hit) {
                hit = j;
                break;
            }
        }

        std::vector<double> alpha(m, 0.0);
        if (hit >= 0) {
            alpha[hit] = 1.0;
            out.push_back(BarycentricStencil{start, std::move(alpha)});
            continue;
        }

        // Barycentric evaluation on shifted nodes (start+j)
        long double den = 0.0L;
        for (int j = 0; j < m; ++j) {
            const long double diff = t - static_cast<long double>(start + j);
            const long double v = wloc[j] / diff;
            tmp[j] = v;
            den += v;
        }

        const long double invden = 1.0L / den;
        long double sumw = 0.0L;
        for (int j = 0; j < m; ++j) {
            alpha_ld[j] = tmp[j] * invden;
            sumw += alpha_ld[j];
        }
        // Renormalize to enforce partition of unity in finite precision.
        if (sumw != 0.0L) {
            const long double invsum = 1.0L / sumw;
            for (int j = 0; j < m; ++j) alpha_ld[j] *= invsum;
        }
        for (int j = 0; j < m; ++j) alpha[j] = static_cast<double>(alpha_ld[j]);

        out.push_back(BarycentricStencil{start, std::move(alpha)});
    }

    return out;
}

std::vector<BarycentricStencil>
compute_index_rational_weights(const std::vector<double>& u_q, std::size_t N_in, int d_in, int m_in) {
    const int N = static_cast<int>(N_in);
    if (N <= 0) return {};

    int d = d_in;
    if (d < 0) d = 0;
    int m = m_in;
    if (m < d + 1) m = d + 1;
    if (m > N) m = N;
    if (m <= 0) return {};
    if (d > m - 1) d = m - 1;

    // Work in a normalized local coordinate to avoid huge intermediate magnitudes:
    //   tloc = t - start in [0, m-1]
    //   qn   = (tloc - c) / s,   yn[j] = (j - c) / s
    // where c=(m-1)/2 and s=(m-1) (or 1 for m=1).
    const long double c = 0.5L * static_cast<long double>(m - 1);
    const long double s = (m > 1) ? static_cast<long double>(m - 1) : 1.0L;
    std::vector<long double> yn(m);
    for (int j = 0; j < m; ++j) yn[j] = (static_cast<long double>(j) - c) / s;

    // Precompute FH node weights on the translation-invariant normalized nodes.
    std::vector<long double> wfh;
    floater_hormann_local_weights(yn, d, wfh);

    // Precompute polynomial barycentric node weights on the same normalized nodes for fallback.
    std::vector<long double> wpoly;
    barycentric_node_weights(yn, wpoly);

    std::vector<BarycentricStencil> out;
    out.reserve(u_q.size());

    const long double tmin = 0.0L;
    const long double tmax = static_cast<long double>(N - 1);
    const long double span = static_cast<long double>(m > 1 ? (m - 1) : 1);
    const long double tol_hit = 64.0L * std::numeric_limits<long double>::epsilon() * span;

    std::vector<long double> tmp(m, 0.0L);
    std::vector<long double> alpha_ld(m, 0.0L);

    for (double uq1 : u_q) {
        long double t = static_cast<long double>(uq1) - 1.0L;
        if (t <= tmin) {
            std::vector<double> alpha(m, 0.0);
            alpha[0] = 1.0;
            out.push_back(BarycentricStencil{0, std::move(alpha)});
            continue;
        }
        if (t >= tmax) {
            const int start = std::max(0, N - m);
            std::vector<double> alpha(m, 0.0);
            alpha[m - 1] = 1.0;
            out.push_back(BarycentricStencil{start, std::move(alpha)});
            continue;
        }

        const int start = pick_uniform_window_start(N, m, t);
        const long double tloc = t - static_cast<long double>(start);
        const long double qn = (tloc - c) / s;

        int hit = -1;
        for (int j = 0; j < m; ++j) {
            const long double node = static_cast<long double>(start + j);
            if (fabsl(t - node) <= tol_hit) {
                hit = j;
                break;
            }
        }
        std::vector<double> alpha(m, 0.0);
        if (hit >= 0) {
            alpha[hit] = 1.0;
            out.push_back(BarycentricStencil{start, std::move(alpha)});
            continue;
        }

        // Evaluate FH weights with compensated summation.
        long double den = 0.0L, comp = 0.0L;
        long double sumabs = 0.0L;
        for (int j = 0; j < m; ++j) {
            const long double diff = qn - yn[j];
            const long double v = wfh[j] / diff;
            tmp[j] = v;
            sumabs += fabsl(v);
            const long double tden = den + v;
            if (fabsl(den) >= fabsl(v)) comp += (den - tden) + v; else comp += (v - tden) + den;
            den = tden;
        }
        den += comp;

        // If catastrophic cancellation produces a near-zero denominator, fall back to
        // polynomial barycentric on the same window (still in normalized coordinates).
        const long double cancel_tol = 1e-14L;
        if (!(den == den) || !std::isfinite(den) || (sumabs > 0.0L && fabsl(den) <= cancel_tol * sumabs)) {
            long double den2 = 0.0L, comp2 = 0.0L;
            for (int jj = 0; jj < m; ++jj) {
                const long double diff = qn - yn[jj];
                const long double v = wpoly[jj] / diff;
                tmp[jj] = v;
                const long double tden = den2 + v;
                if (fabsl(den2) >= fabsl(v)) comp2 += (den2 - tden) + v; else comp2 += (v - tden) + den2;
                den2 = tden;
            }
            den2 += comp2;
            const long double invden2 = 1.0L / den2;
            long double sumw = 0.0L;
            for (int jj = 0; jj < m; ++jj) {
                alpha_ld[jj] = tmp[jj] * invden2;
                sumw += alpha_ld[jj];
            }
            if (sumw != 0.0L) {
                const long double invsum = 1.0L / sumw;
                for (int jj = 0; jj < m; ++jj) alpha_ld[jj] *= invsum;
            }
            for (int jj = 0; jj < m; ++jj) alpha[jj] = static_cast<double>(alpha_ld[jj]);
            out.push_back(BarycentricStencil{start, std::move(alpha)});
            continue;
        }

        const long double invden = 1.0L / den;
        long double sumw = 0.0L;
        for (int j = 0; j < m; ++j) {
            alpha_ld[j] = tmp[j] * invden;
            sumw += alpha_ld[j];
        }
        if (sumw != 0.0L) {
            const long double invsum = 1.0L / sumw;
            for (int j = 0; j < m; ++j) alpha_ld[j] *= invsum;
        }
        for (int j = 0; j < m; ++j) alpha[j] = static_cast<double>(alpha_ld[j]);

        out.push_back(BarycentricStencil{start, std::move(alpha)});
    }

    return out;
}

std::vector<BarycentricStencil>
compute_index_hermite_weights(const std::vector<double>& u_q, std::size_t N_in, int order_in) {
    const int N = static_cast<int>(N_in);
    if (N <= 0) return {};

    int n = order_in;
    if (n < 1) n = 1;
    if (n > N - 1) n = N - 1;
    const int m = n + 1;
    if (m < 2) {
        return compute_index_poly_weights(u_q, N_in, 0);
    }
    if (N < m) {
        return compute_index_poly_weights(u_q, N_in, std::min(n, N - 1));
    }

    // Precompute Lagrange derivative weights at each node of the local stencil.
    // Nodes are uniform integers 0..m-1 (shifted later by `start`).
    std::vector<long double> nodes(m);
    for (int j = 0; j < m; ++j) nodes[j] = static_cast<long double>(j);
    std::vector<long double> wloc;
    barycentric_node_weights(nodes, wloc);

    // D[k*m + j] = L_j'(x_k) where x_k = k.
    std::vector<long double> D((std::size_t)m * m, 0.0L);
    for (int k = 0; k < m; ++k) {
        long double sum = 0.0L;
        for (int j = 0; j < m; ++j) {
            if (j == k) continue;
            const long double diff = static_cast<long double>(k - j);
            const long double v = wloc[j] / (wloc[k] * diff);
            D[(std::size_t)k * m + j] = v;
            sum += v;
        }
        D[(std::size_t)k * m + k] = -sum;
    }
    std::vector<BarycentricStencil> out;
    out.reserve(u_q.size());

    const long double tmin = 0.0L;
    const long double tmax = static_cast<long double>(N - 1);
    const long double tol_hit = 64.0L * std::numeric_limits<long double>::epsilon();

    for (double uq1 : u_q) {
        // 1-based fractional index u in [1,N] -> 0-based coordinate t in [0,N-1]
        long double t = static_cast<long double>(uq1) - 1.0L;

        if (t <= tmin) {
            std::vector<double> alpha(m, 0.0);
            alpha[0] = 1.0;
            out.push_back(BarycentricStencil{0, std::move(alpha)});
            continue;
        }
        if (t >= tmax) {
            const int start = N - m;
            std::vector<double> alpha(m, 0.0);
            alpha[m - 1] = 1.0;
            out.push_back(BarycentricStencil{start, std::move(alpha)});
            continue;
        }

        // Segment index i such that t in [i, i+1)
        int i = static_cast<int>(std::floor(t));
        if (i < 0) i = 0;
        if (i > N - 2) i = N - 2;

        // Exact hit: return delta if very close to an integer node.
        {
            const long double node = static_cast<long double>(i);
            const long double nodep = static_cast<long double>(i + 1);
            if (fabsl(t - node) <= tol_hit) {
                const int start = clamp_int(i - 1, 0, N - m);
                std::vector<double> alpha(m, 0.0);
                const int local = i - start;
                if (local >= 0 && local < m) alpha[local] = 1.0;
                out.push_back(BarycentricStencil{start, std::move(alpha)});
                continue;
            }
            if (fabsl(t - nodep) <= tol_hit) {
                const int start = clamp_int(i - 1, 0, N - m);
                std::vector<double> alpha(m, 0.0);
                const int local = (i + 1) - start;
                if (local >= 0 && local < m) alpha[local] = 1.0;
                out.push_back(BarycentricStencil{start, std::move(alpha)});
                continue;
            }
        }

        const long double s = t - static_cast<long double>(i); // in (0,1)

        // Hermite basis on [0,1]
        const long double s2 = s * s;
        const long double s3 = s2 * s;
        const long double h00 = (2.0L * s3 - 3.0L * s2 + 1.0L);
        const long double h10 = (s3 - 2.0L * s2 + s);
        const long double h01 = (-2.0L * s3 + 3.0L * s2);
        const long double h11 = (s3 - s2);

        // Choose a local stencil of size m that always contains i and i+1.
        // Center it around the segment.
        const int half_left = (m - 2) / 2;
        const int start = clamp_int(i - half_left, 0, N - m);
        const int r0 = i - start;
        const int r1 = r0 + 1;

        std::vector<long double> alpha_ld(m, 0.0L);

        // Value contributions at endpoints (y_i and y_{i+1}).
        if (r0 >= 0 && r0 < m) alpha_ld[r0] += h00;
        if (r1 >= 0 && r1 < m) alpha_ld[r1] += h01;

        // Slope contributions: m_i = sum_j D[r0,j] y_{start+j}, m_{i+1} = sum_j D[r1,j] y_{start+j}
        // Thus alpha[j] += h10*D[r0,j] + h11*D[r1,j].
        for (int j = 0; j < m; ++j) {
            alpha_ld[j] += h10 * D[(std::size_t)r0 * m + j] + h11 * D[(std::size_t)r1 * m + j];
        }

        std::vector<double> alpha(m, 0.0);
        for (int j = 0; j < m; ++j) alpha[j] = static_cast<double>(alpha_ld[j]);

        // Enforce partition of unity in finite precision (constant reproduction).
        long double sum = 0.0L;
        for (int j = 0; j < m; ++j) sum += static_cast<long double>(alpha[j]);
        if (sum != 0.0L) {
            const double invsum = 1.0 / static_cast<double>(sum);
            for (int j = 0; j < m; ++j) alpha[j] *= invsum;
        }

        out.push_back(BarycentricStencil{start, std::move(alpha)});
    }

    return out;
}

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
        // Choose a near-optimal window of size m (minimize spread)
        int start = pick_window_start(x, m, q);
        xn.clear(); xn.reserve(m);
        for (int j = 0; j < m; ++j) xn.push_back(x[start + j]);

    // Exact hit (with tolerance relative to true local span; avoid over-large tol)
    long double span = xn.back() - xn.front();
    long double tol = 64 * std::numeric_limits<long double>::epsilon() * (span > 0.0L ? span : 1.0L);
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

        // Normalize window to reduce scale disparities: y = (x - c) / s
    long double a = xn.front();
    long double b = xn.back();
        long double c = 0.5L * (a + b);
    long double s = b - a;
    if (s <= 0.0L) s = 1.0L; // degenerate safety; otherwise use true span
        std::vector<long double> yn(m);
        for (int j = 0; j < m; ++j) yn[j] = (xn[j] - c) / s;
        long double qn = (q - c) / s;

        // Local FH weights on normalized nodes
        floater_hormann_local_weights(yn, d, wfh);

        // Evaluate FH with Neumaier compensated sum
        tmp.assign(m, 0.0L);
        long double denFH = 0.0L, compFH = 0.0L, sumabsFH = 0.0L;
        for (int j = 0; j < m; ++j) {
            long double v = wfh[j] / (qn - yn[j]);
            tmp[j] = v; // reuse tmp for FH v_j
            sumabsFH += fabsl(v);
            long double t = denFH + v;
            if (fabsl(denFH) >= fabsl(v)) compFH += (denFH - t) + v; else compFH += (v - t) + denFH;
            denFH = t;
        }
        denFH += compFH;
        std::vector<double> alphaFH(m, 0.0), alphaPoly(m, 0.0);
        long double lebesgueFH = std::numeric_limits<long double>::infinity();
        if ((denFH == denFH) && std::isfinite(denFH)) {
            long double invden = 1.0L / denFH;
            lebesgueFH = 0.0L;
            for (int j = 0; j < m; ++j) {
                double aj = (double)(tmp[j] * invden);
                alphaFH[j] = aj;
                lebesgueFH += std::fabs(aj);
            }
        }

        // Compute polynomial barycentric on same normalized window (reference, also compensated)
        std::vector<long double> wloc;
        barycentric_node_weights(yn, wloc);
        long double denPL = 0.0L, compPL = 0.0L;
        for (int j = 0; j < m; ++j) {
            long double v = wloc[j] / (qn - yn[j]);
            // overwrite tmp to reuse memory, but after FH alpha already formed
            tmp[j] = v;
            long double t = denPL + v;
            if (fabsl(denPL) >= fabsl(v)) compPL += (denPL - t) + v; else compPL += (v - t) + denPL;
            denPL = t;
        }
        denPL += compPL;
        long double lebesguePL = std::numeric_limits<long double>::infinity();
        if ((denPL == denPL) && std::isfinite(denPL)) {
            long double invden2 = 1.0L / denPL;
            lebesguePL = 0.0L;
            for (int j = 0; j < m; ++j) {
                double aj = (double)(tmp[j] * invden2);
                alphaPoly[j] = aj;
                lebesguePL += std::fabs(aj);
            }
        }

        // Decide: use polynomial if FH is ill-conditioned or clearly worse by Lebesgue sum
        const long double cancel_tol = 1e-10L; // fallback when |den| << sum of terms
        bool fh_bad = !(denFH == denFH) || !std::isfinite(denFH) || (sumabsFH > 0.0L && fabsl(denFH) <= cancel_tol * sumabsFH);
        bool take_poly = fh_bad;
        if (!take_poly) {
            if (lebesguePL < std::numeric_limits<long double>::infinity() && lebesgueFH < std::numeric_limits<long double>::infinity()) {
                // Prefer the weights with smaller Lebesgue sum; require a margin to avoid flip-flop
                if (lebesgueFH > 2.0L * lebesguePL) take_poly = true;
            }
        }

        if (take_poly) {
            alpha = std::move(alphaPoly);
        } else {
            alpha = std::move(alphaFH);
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
