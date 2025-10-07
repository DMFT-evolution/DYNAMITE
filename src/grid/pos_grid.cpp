#include "grid/pos_grid.hpp"
#include "grid/theta_grid.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace {
// Local barycentric polynomial interpolation.
// Precomputes weights for a small per-interval stencil; evaluation is O(m).
// Accurate and stable on dense, smooth, monotone, nonuniform grids.
struct LocalBarycentric {
	std::vector<double> x; // strictly increasing knots
	std::vector<double> y; // values at knots
	int m = 8;             // stencil size (polynomial degree m-1)

	// For interval i in [0,n-2], choose left index L(i) so nodes are x[L..L+m-1].
	std::vector<int> left_of_interval; // size n-1

	// For left index L in [0,n-m], store barycentric weights for x[L..L+m-1].
	std::vector<double> weights; // flat: (n-m+1) blocks of m

	static int clamp_int(int v, int lo, int hi) {
		return v < lo ? lo : (v > hi ? hi : v);
	}

	void build(const std::vector<double>& xin, const std::vector<double>& yin, int m_in = 8) {
		const std::size_t n = xin.size();
		if (n < 2 || yin.size() != n) throw std::invalid_argument("LocalBarycentric: bad input size");
		x = xin; y = yin;
		// Check strict monotonicity
		for (std::size_t i = 1; i < n; ++i) {
			if (!(x[i] > x[i-1])) throw std::invalid_argument("LocalBarycentric: x must be strictly increasing");
		}

		// Stencil size from env or argument
		int m_env = m_in;
		if (const char* env = std::getenv("DMFE_POS_BARY_M")) {
			try { m_env = std::max(2, std::min(32, int(std::strtol(env, nullptr, 10)))); } catch (...) {}
		}
		// Constrain to [4,16] and <= n
		m = std::max(4, std::min(16, m_env));
		if (m > int(n)) m = int(n);
		if (m < 2) m = 2;

		const int Nm = int(n) - m;
		left_of_interval.resize(n - 1);
		// Center the window on the interval
		for (std::size_t i = 0; i < n - 1; ++i) {
			int center = int(i); // interval [i, i+1]
			int L = clamp_int(center - m/2 + 1, 0, std::max(0, Nm));
			left_of_interval[i] = L;
		}

		// Precompute weights for each possible left index L
		const int nLeft = std::max(0, Nm) + 1;
		weights.assign(std::size_t(nLeft) * std::size_t(m), 0.0);
		for (int L = 0; L < nLeft; ++L) {
			// Standard barycentric weights for x[L..L+m-1]
			for (int j = 0; j < m; ++j) {
				double w = 1.0;
				const double xj = x[std::size_t(L + j)];
				for (int k = 0; k < m; ++k) {
					if (k == j) continue;
					const double diff = xj - x[std::size_t(L + k)];
					w *= diff;
				}
				weights[std::size_t(L) * std::size_t(m) + std::size_t(j)] = 1.0 / w;
			}
		}
	}

	inline double eval(double xi) const {
		const std::size_t n = x.size();
		if (xi <= x.front()) return y.front();
		if (xi >= x.back()) return y.back();

		// Find containing interval
		auto it = std::upper_bound(x.begin(), x.end(), xi);
		std::size_t i = std::max<std::size_t>(1, std::size_t(it - x.begin())) - 1; // in [0, n-2]
		const int L = left_of_interval[i];
		const int nLeft = std::max(0, int(n) - m) + 1;
		const int Lclamped = clamp_int(L, 0, nLeft - 1);

		const double* wptr = &weights[std::size_t(Lclamped) * std::size_t(m)];

		// Exact hit avoids division by zero
		for (int j = 0; j < m; ++j) {
			const std::size_t idx = std::size_t(Lclamped + j);
			if (xi == x[idx]) return y[idx];
		}

		double num = 0.0, den = 0.0;
		for (int j = 0; j < m; ++j) {
			const std::size_t idx = std::size_t(Lclamped + j);
			const double diff = xi - x[idx];
			const double w = wptr[j] / diff;
			den += w;
			num += w * y[idx];
		}
		return num / den;
	}
};

} // namespace

void generate_pos_grids(std::size_t len,
						double Tmax,
						const std::vector<double>& theta,
						const std::vector<double>& phi1,
						const std::vector<double>& phi2,
						std::vector<double>& posA1y,
						std::vector<double>& posA2y,
						std::vector<double>& posB2y) {
	const std::size_t N = theta.size();
	if (N != len) throw std::invalid_argument("generate_pos_grids: theta size must match len");
	if (phi1.size() != N*N || phi2.size() != N*N) throw std::invalid_argument("generate_pos_grids: phi sizes must be N*N");

	// Build inverse map θ -> y:
	// - Densely oversample analytical θ(y) via theta_of_vec (vectorized)
	// - Enforce strict monotonicity in sampled θ
	// - Fit a local barycentric interpolant for θ -> y
	// Oversample factor defaults to 20; override with DMFE_POS_OVERSAMPLE.
	std::size_t oversample = 20;
	if (const char* env = std::getenv("DMFE_POS_OVERSAMPLE")) {
		try {
			long v = std::strtol(env, nullptr, 10);
			if (v > 1) oversample = static_cast<std::size_t>(v);
		} catch (...) {}
	}

	// Step 1: Densely sample analytical θ(y) (vectorized)
	const std::size_t M = N * oversample;
	std::vector<double> inv_x, inv_y;  // inv_x = theta values, inv_y = y positions
	inv_x.reserve(M);
	inv_y.reserve(M);
	
	// Build indices vector for vectorized theta computation
	std::vector<double> indices(M);
	for (std::size_t k = 0; k < M; ++k) {
		double y = 1.0 + (static_cast<double>(N) - 1.0) * (double(k) / double(M - 1));
		indices[k] = y - 1.0;  // Convert to 0-based index
		inv_y.push_back(y);
	}
	
	// Compute all θ values at once (vectorized; reuses internal constants)
	theta_of_vec(indices, N, Tmax, inv_x);
	
	// Ensure strict monotonicity in sampled theta
	for (std::size_t k = 1; k < M; ++k) {
		if (inv_x[k] <= inv_x[k-1]) {
			inv_x[k] = std::nextafter(inv_x[k-1], std::numeric_limits<double>::infinity());
		}
	}
	
	// Step 2: Build inverse interpolant θ -> y using local barycentric
	LocalBarycentric inv_interp;
	inv_interp.build(inv_x, inv_y, 12); // default to a larger stencil for near-ulp accuracy

	posA1y.assign(N * N, 0.0);
	posA2y.assign(N * N, 0.0);
	posB2y.assign(N * N, 0.0);
	const double tiny = 1e-200; // epsilon for stability in B2

	// Precompute domain bounds for safe evaluation
	const double theta_min = inv_x.front();
	const double theta_max = inv_x.back();
	auto safe_eval = [&](double val) {
		if (val <= theta_min) return inv_y.front();
		if (val >= theta_max) return inv_y.back();
		return inv_interp.eval(val);
	};

	for (std::size_t i = 0; i < N; ++i) {
		const double ti = theta[i];
		for (std::size_t j = 0; j < N; ++j) {
			const double v1 = phi1[i * N + j];
			const double v2 = phi2[i * N + j];
			
			posA1y[i * N + j] = safe_eval(v1);
			posA2y[i * N + j] = safe_eval(v2);
			
			// posB2y = inv_theta[theta[j] / (phi2 - 10^-200)]
			const double denom = v2 - tiny;
			double argB2;
			if (std::abs(denom) < 1e-300) {
				// Near-zero denominator: clamp to domain boundary
				argB2 = (ti > 0) ? theta_max : theta_min;
			} else {
				argB2 = ti / denom;
			}
			posB2y[i * N + j] = safe_eval(argB2);
		}
	}
}

