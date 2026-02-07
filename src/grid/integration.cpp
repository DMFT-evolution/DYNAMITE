#include "grid/integration.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "grid/grid_io.hpp"
#include "grid/theta_grid.hpp"

namespace {

// ---------------- Spline-consistent integration helpers ----------------

// Open (clamped) knot vector using x as parameter values; endpoints repeated p+1 times.
static std::vector<long double>
build_open_knot_vector(const std::vector<long double>& x, int p) {
    const int N = static_cast<int>(x.size());
    if (N < p + 1) throw std::invalid_argument("spline integration: need N >= p+1");
    const int mBasis = N; // number of basis functions equals number of data points
    const int K = mBasis + p + 1; // number of knots
    std::vector<long double> t(K);
    for (int i = 0; i <= p; ++i) t[i] = x.front();
    for (int i = 0; i <= p; ++i) t[K - 1 - i] = x.back();
    const int n = N - 1;
    for (int j = p + 1; j <= n; ++j) {
        long double sum = 0.0L;
        for (int i = j - p; i <= j - 1; ++i) sum += x[i];
        t[j] = sum / static_cast<long double>(p);
    }
    return t;
}

// Find span such that t[span] <= u < t[span+1]
static int find_span(int mBasis, int p, long double u, const std::vector<long double>& t) {
    const int n = mBasis - 1;
    if (u >= t[n + 1]) return n;
    if (u <= t[p]) return p;
    int low = p, high = n + 1, mid = (low + high) / 2;
    while (!(u >= t[mid] && u < t[mid + 1])) {
        if (u < t[mid]) high = mid; else low = mid;
        mid = (low + high) / 2;
    }
    return mid;
}

// Basis functions N_{i-p..i,p}(u). Returns N size p+1.
static void basis_funs(int span, long double u, int p, const std::vector<long double>& t,
                       std::vector<long double>& N) {
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

// Assemble banded collocation matrix A (N x N), where A_ij = N_j(x_i).
// Only fill band entries where |i-j| <= p.
static void assemble_collocation(const std::vector<long double>& x, int p,
                                 const std::vector<long double>& knots,
                                 std::vector<long double>& A) {
    const int N = static_cast<int>(x.size());
    A.assign(static_cast<std::size_t>(N) * N, 0.0L);
    const int mBasis = N;
    std::vector<long double> Nvals(p + 1);
    for (int i = 0; i < N; ++i) {
        long double u = x[i];
        int span = find_span(mBasis, p, u, knots);
        basis_funs(span, u, p, knots, Nvals);
        int first = span - p;
        for (int j = 0; j <= p; ++j) {
            int col = first + j;
            if (col >= 0 && col < N) A[static_cast<std::size_t>(i) * N + col] = Nvals[j];
        }
    }
}

// Simple banded LU factorization without pivoting on A (in-place), band half-width p.
// A is full row-major N x N but only band entries are assumed non-zero.
static void banded_lu(std::vector<long double>& A, int N, int p) {
    for (int k = 0; k < N; ++k) {
        long double akk = A[static_cast<std::size_t>(k) * N + k];
        // Assume non-singular; no pivoting
        for (int i = k + 1; i <= std::min(N - 1, k + p); ++i) {
            long double lik = A[static_cast<std::size_t>(i) * N + k] /= akk;
            for (int j = k + 1; j <= std::min(N - 1, k + p); ++j) {
                A[static_cast<std::size_t>(i) * N + j] -= lik * A[static_cast<std::size_t>(k) * N + j];
            }
        }
    }
}

// Solve A x = b using LU factors in A (unit-lower L and upper U embedded), band p.
[[maybe_unused]] static void banded_lu_solve(const std::vector<long double>& A, int N, int p,
                            const std::vector<long double>& b,
                            std::vector<long double>& x) {
    x = b;
    // Forward: L y = b
    for (int i = 0; i < N; ++i) {
        int j0 = std::max(0, i - p);
        for (int j = j0; j < i; ++j) {
            x[i] -= A[static_cast<std::size_t>(i) * N + j] * x[j];
        }
    }
    // Backward: U x = y
    for (int i = N - 1; i >= 0; --i) {
        int j1 = std::min(N - 1, i + p);
        for (int j = i + 1; j <= j1; ++j) {
            x[i] -= A[static_cast<std::size_t>(i) * N + j] * x[j];
        }
        x[i] /= A[static_cast<std::size_t>(i) * N + i];
    }
}

// Solve A^T x = b using LU factors (no pivoting). Use transposed triangular solves.
static void banded_lu_solve_transpose(const std::vector<long double>& A, int N, int p,
                                      const std::vector<long double>& b,
                                      std::vector<long double>& x) {
    x = b;
    // Solve U^T y = b (U upper) -> forward-like
    for (int i = 0; i < N; ++i) {
        int j0 = std::max(0, i - p);
        for (int j = j0; j < i; ++j) {
            // U^T has (j,i) = U(i,j)
            x[i] -= A[static_cast<std::size_t>(j) * N + i] * x[j];
        }
        x[i] /= A[static_cast<std::size_t>(i) * N + i];
    }
    // Solve L^T x = y (L unit-lower) -> backward-like
    for (int i = N - 1; i >= 0; --i) {
        int j1 = std::min(N - 1, i + p);
        for (int j = i + 1; j <= j1; ++j) {
            // L^T has (i,j) = L(j,i)
            x[i] -= A[static_cast<std::size_t>(j) * N + i] * x[j];
        }
        // No division: diagonal of L is 1
    }
}

// Compute integral of each B-spline basis over [a,b]: M_j = (t_{j+p+1} - t_j)/(p+1)
static std::vector<long double>
compute_basis_integrals(const std::vector<long double>& knots, int N, int p) {
    std::vector<long double> M(N, 0.0L);
    for (int j = 0; j < N; ++j) {
        long double num = knots[j + p + 1] - knots[j];
        M[j] = num / static_cast<long double>(p + 1);
    }
    return M;
}

// (No mapping needed anymore) Degree p will be taken directly from `order` and
// clamped into [1, N-1] at call site.

// Fixed Gauss-Legendre rule on [-1,1] (16-point) for stable per-interval quadrature.
static constexpr int GL16 = 16;
static const long double gl16_x[GL16] = {
    -0.989400934991649932596154173450L,
    -0.944575023073232576077988415535L,
    -0.865631202387831743880467897712L,
    -0.755404408355003033895101194847L,
    -0.617876244402643748446671764049L,
    -0.458016777657227386342419442984L,
    -0.281603550779258913230460501460L,
    -0.095012509837637440185319335425L,
     0.095012509837637440185319335425L,
     0.281603550779258913230460501460L,
     0.458016777657227386342419442984L,
     0.617876244402643748446671764049L,
     0.755404408355003033895101194847L,
     0.865631202387831743880467897712L,
     0.944575023073232576077988415535L,
     0.989400934991649932596154173450L
};
static const long double gl16_w[GL16] = {
    0.027152459411754094851780572456L,
    0.062253523938647892862843836994L,
    0.095158511682492784809925107602L,
    0.124628971255533872052476282192L,
    0.149595988816576732081501730547L,
    0.169156519395002538189312079030L,
    0.182603415044923588866763667969L,
    0.189450610455068496285396723208L,
    0.189450610455068496285396723208L,
    0.182603415044923588866763667969L,
    0.169156519395002538189312079030L,
    0.149595988816576732081501730547L,
    0.124628971255533872052476282192L,
    0.095158511682492784809925107602L,
    0.062253523938647892862843836994L,
    0.027152459411754094851780572456L
};

} // namespace

void compute_integration_weights(const std::vector<long double>& theta,
                                 int order,
                                 std::vector<long double>& w) {
    const int N = static_cast<int>(theta.size());
    w.clear();
    if (N == 0) return;

    // Take spline degree directly from `order`; clamp to feasible range [1, N-1]
    int p = order;
    if (p < 1) p = 1;
    if (p > N - 1) p = std::max(1, N - 1);

    // Build clamped knot vector and banded collocation matrix
    auto knots = build_open_knot_vector(theta, p);
    std::vector<long double> A; A.reserve(static_cast<std::size_t>(N) * std::min(N, 2*p+1));
    assemble_collocation(theta, p, knots, A);

    // Factor banded LU (no pivoting)
    banded_lu(A, N, p);

    // Build basis integrals vector M (exact)
    std::vector<long double> M = compute_basis_integrals(knots, N, p);

    // Solve A^T w = M
    std::vector<long double> wl;
    banded_lu_solve_transpose(A, N, p, M, wl);

    // Output
    w.resize(N);
    for (int i = 0; i < N; ++i) w[i] = wl[i];
}

void compute_integration_weights_index_spline(std::size_t len,
                                              double Tmax,
                                              int order,
                                              std::vector<long double>& w,
                                              double alpha,
                                              double delta) {
    const int N = static_cast<int>(len);
    w.clear();
    if (N <= 0) return;

    int p = order;
    if (p < 1) p = 1;
    if (p > N - 1) p = std::max(1, N - 1);

    // Uniform index nodes x_i = i (0..N-1)
    std::vector<long double> x(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) x[static_cast<std::size_t>(i)] = static_cast<long double>(i);

    // Build knot vector and collocation matrix on the uniform index grid.
    auto knots = build_open_knot_vector(x, p);
    std::vector<long double> A;
    assemble_collocation(x, p, knots, A);
    banded_lu(A, N, p);

    // Build RHS moments M_j = \int B_j(u) * theta'(u) du using per-interval Gauss-Legendre.
    // We evaluate theta'(u) analytically from the known theta mapping.
    std::vector<long double> M(static_cast<std::size_t>(N), 0.0L);
    std::vector<long double> Nvals(p + 1);

    // Precompute theta'(u) at all quadrature points to avoid repeating expensive setup.
    const int Q = GL16;
    const std::size_t nIntervals = (N >= 2) ? static_cast<std::size_t>(N - 1) : 0;
    std::vector<double> uq;
    uq.reserve(nIntervals * static_cast<std::size_t>(Q));
    for (int k = 0; k < N - 1; ++k) {
        const long double a = static_cast<long double>(k);
        const long double mid = a + 0.5L;
        for (int q = 0; q < Q; ++q) {
            const long double u = mid + 0.5L * gl16_x[q];
            uq.push_back(static_cast<double>(u));
        }
    }
    std::vector<long double> dtheta;
    theta_prime_of_vec(uq, len, Tmax, dtheta, alpha, delta);

    // Accumulate M by sweeping intervals.
    std::size_t idx = 0;
    for (int k = 0; k < N - 1; ++k) {
        const long double mid = static_cast<long double>(k) + 0.5L;
        for (int q = 0; q < Q; ++q, ++idx) {
            const long double u = mid + 0.5L * gl16_x[q];
            const long double wt = 0.5L * gl16_w[q]; // scale from [-1,1] to [k,k+1]
            const long double dth = (idx < dtheta.size()) ? dtheta[idx] : 0.0L;
            if (dth == 0.0L) continue;

            const int span = find_span(N, p, u, knots);
            basis_funs(span, u, p, knots, Nvals);
            const int first = span - p;
            for (int j = 0; j <= p; ++j) {
                const int col = first + j;
                if (col >= 0 && col < N) {
                    M[static_cast<std::size_t>(col)] += Nvals[j] * dth * wt;
                }
            }
        }
    }

    // Solve A^T w = M
    std::vector<long double> wl;
    banded_lu_solve_transpose(A, N, p, M, wl);

    w.resize(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) w[static_cast<std::size_t>(i)] = wl[static_cast<std::size_t>(i)];

    // Optional: enforce constant reproduction in finite precision by renormalizing.
    // For theta in [0,1], integral of 1 is 1, so sum(w) should be ~1.
    long double sum = 0.0L;
    for (int i = 0; i < N; ++i) sum += w[static_cast<std::size_t>(i)];
    if (sum != 0.0L && std::isfinite(sum)) {
        const long double invsum = 1.0L / sum;
        for (int i = 0; i < N; ++i) w[static_cast<std::size_t>(i)] *= invsum;
    }
}

// Backward-compatible overload for double theta
void compute_integration_weights(const std::vector<double>& theta,
                                 int order,
                                 std::vector<long double>& w) {
    std::vector<long double> thetal(theta.size());
    for (std::size_t i = 0; i < theta.size(); ++i) thetal[i] = static_cast<long double>(theta[i]);
    compute_integration_weights(thetal, order, w);
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
