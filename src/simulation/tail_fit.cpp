#include "simulation/tail_fit.hpp"
#include "simulation/simulation_data.hpp"
#include "core/config.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <array>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

extern SimulationConfig config;
extern SimulationData* sim;
extern int last_rollback_loop; // Track rollbacks to reset smoothing

// Helper: compute best constant fit mean and SSE for window [i, i+W)
static inline void window_mean_sse(const double* y, size_t i, size_t W, double& mean, double& sse) {
	double sum = 0.0;
#pragma omp simd reduction(+:sum)
	for (size_t k = 0; k < W; ++k) sum += y[i + k];
	mean = sum / static_cast<double>(W);
	double err = 0.0;
#pragma omp simd reduction(+:err)
	for (size_t k = 0; k < W; ++k) {
		double d = y[i + k] - mean;
		err += d * d;
	}
	sse = err;
#if DMFE_TAIL_FIT_DEBUG
	if (config.debug) {
#ifdef _OPENMP
#pragma omp critical
#endif
		{
			std::cout << "[tail_fit] window_start_off=" << i
					  << ", sum=" << sum
					  << ", sse=" << sse << std::endl;
		}
	}
#endif
}

void tailFitBlendCPU() {
	const size_t len = config.len;
	if (len < 32) return; // not enough points to bother
	// Operate on last time-slice of QKv
	const size_t t1len = sim->h_t1grid.size();
	if (t1len == 0) return;
	const size_t base = (t1len - 1) * len;

	// y_j = (1 - QK)/(1 - theta)^2, only valid where theta<1 and not too close to 1 to avoid div overflow; but we want the tail near 1
	// We'll use the second half of theta indices [start, len)
	const size_t start = len / 2;
	const size_t W = 16; // window size for local statistics
	if (len - start < W) return;

	// Reuse/resize temp buffer
	std::vector<double>& tmp = sim->h_temp0;
	tmp.resize(len - start);

	// Precompute y (parallelizable)
#pragma omp parallel for
	for (size_t j = start; j < len; ++j) {
		double th = sim->h_theta[j];
		double one_m = 1.0 - th;
		double denom = one_m * one_m;
		double qk = sim->h_QKv[base + j];
		double y = (1.0 - qk) / (denom > 0 ? denom : 1e-300);
		tmp[j - start] = y;
	}

	// Persistent best window offset across time steps
	static size_t prev_best_off = static_cast<size_t>(-1); // SIZE_MAX sentinel
	static size_t prev_t1len    = 0;
	// Smoothed alpha state
	static double alpha_smoothed = 0.0;
	static int last_known_rollback = -1000; // Track last rollback we've seen

	const size_t N = tmp.size();
	const bool new_time_step = (prev_t1len != t1len); // t1len increases when a new slice appended

	size_t best_off = 0;
	double best_sse = std::numeric_limits<double>::infinity();
	double best_mean = 0.0;

	// Determine candidate windows: initial full scan or restricted scan based on previous best
	if (prev_best_off == static_cast<size_t>(-1) || !new_time_step) {
		// First invocation OR same time step re-entry: full scan
		const size_t window_count = (N - W) / W + 1;
		std::vector<double> means(window_count), sses(window_count);
#pragma omp parallel for
		for (long w = 0; w < static_cast<long>(window_count); ++w) {
			size_t off = static_cast<size_t>(w) * W;
			double mean, sse;
			window_mean_sse(tmp.data(), off, W, mean, sse);
			means[w] = mean;
			sses[w]  = sse;
		}
		for (size_t w = 0; w < window_count; ++w) {
			if (sses[w] < best_sse) {
				best_sse = sses[w];
				best_off = w * W;
				best_mean = means[w];
			}
		}
	} else {
		// Restricted scan: previous best window and its immediate left/right neighbors (shifted by W)
	std::vector<size_t> candidates;
	candidates.push_back(prev_best_off);
	if (prev_best_off >= W) candidates.push_back(prev_best_off - W);
	if (prev_best_off + W + W <= N) candidates.push_back(prev_best_off + W);
	std::sort(candidates.begin(), candidates.end());
	candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
	for (size_t off : candidates) {
			if (off + W > N) continue;
			double mean, sse;
			window_mean_sse(tmp.data(), off, W, mean, sse);
			if (sse < best_sse) { best_sse = sse; best_off = off; best_mean = mean; }
		}
	}

	prev_best_off = best_off; // remember for next time step
	prev_t1len    = t1len;

	// Blend strategy: compute alpha = best_mean, then replace QK for indices j >= (start + best_off)
	// with a smooth ramp over R points up to pure fit.
	const double alpha_raw = best_mean;
	const size_t j0 = start + best_off;

	// Estimate variance & standard error for confidence-based smoothing
	double var = (W > 1) ? (best_sse / static_cast<double>(W - 1)) : 0.0;
	double stderr_mean = (W > 0) ? std::sqrt(std::max(var, 0.0) / static_cast<double>(W)) : 0.0;

	// Detect rollback: if last_rollback_loop changed, reset smoothing to raw value
	bool rollback_detected = (last_rollback_loop != last_known_rollback);
	if (rollback_detected) {
		last_known_rollback = last_rollback_loop;
		if (alpha_raw != 0.0) {
			alpha_smoothed = alpha_raw;
#if DMFE_TAIL_FIT_DEBUG
			if (config.debug) {
				std::cout << "[tail_fit] Rollback detected (loop=" << last_rollback_loop 
						  << "), resetting alpha_smooth=" << alpha_raw << std::endl;
			}
#endif
		}
	}

	// Adaptive smoothing factor in [min_gain, max_gain]
	// Higher relative error -> smaller gain (slower adaptation, more smoothing)
	const double max_gain = 0.50;
	const double min_gain = 0.10;
	double rel_err = (std::abs(alpha_raw) > 1e-12) ? stderr_mean / std::abs(alpha_raw) : 1.0;
	double gain = max_gain / (1.0 + 2.0 * rel_err);
	gain = std::max(min_gain, std::min(gain, max_gain));

	// Initialize or update smoothed alpha
	if (!(alpha_smoothed == alpha_smoothed) || (alpha_raw == 0.0 && rollback_detected)) {
		// Initialize if NaN or if rollback with zero raw value
		if (alpha_raw != 0.0) {
			alpha_smoothed = alpha_raw;
		} else if (!(alpha_smoothed == alpha_smoothed)) {
			alpha_smoothed = 0.0;
		}
		// else: keep existing alpha_smoothed if raw is zero but we have a valid smoothed value
	} else if (alpha_raw == 0.0) {
		// If raw is zero but smoothed is valid, keep smoothed (don't update)
		// This allows continuation with previous valid smoothed value
	} else {
		// Normal case: apply exponential moving average
		alpha_smoothed = (1.0 - gain) * alpha_smoothed + gain * alpha_raw;
	}
	const double alpha = alpha_smoothed;

	// Compute residuals between fit and numeric over tail region [j0, len)
	std::vector<double> residuals(len - j0);
#pragma omp parallel for
	for (size_t j = j0; j < len; ++j) {
		double th = sim->h_theta[j];
		double one_m = 1.0 - th;
		double qk_fit = 1.0 - alpha * one_m * one_m;
		double qk_num = sim->h_QKv[base + j];
		residuals[j - j0] = qk_fit - qk_num;
	}

	// Compute mean absolute residual (measure of fit error)
	double mean_abs_residual = 0.0;
	for (double r : residuals) mean_abs_residual += std::fabs(r);
	mean_abs_residual /= static_cast<double>(residuals.size());

	// Estimate noise level from residuals std dev
	double mean_residual = 0.0;
	for (double r : residuals) mean_residual += r;
	mean_residual /= static_cast<double>(residuals.size());
	double residual_variance = 0.0;
	for (double r : residuals) {
		double d = r - mean_residual;
		residual_variance += d * d;
	}
	residual_variance /= std::max(1.0, static_cast<double>(residuals.size()) - 1.0);
	double residual_std = std::sqrt(std::max(0.0, residual_variance));

	// Gate: apply if mean absolute residual is less than 3*sigma (noise-based relative criterion)
	// This adapts to local noise level and allows larger deviations when window switches
	double noise_threshold = 3.0 * residual_std;
	bool fit_quality_ok = (mean_abs_residual <= noise_threshold);
	
	// Also require SSE to be reasonable (not wildly fluctuating)
	double sseTol = std::sqrt(config.delta_max);
	bool sse_ok = (best_sse <= sseTol);
	
	// Don't apply if alpha is exactly zero (no meaningful fit)
	bool alpha_nonzero = (std::abs(alpha) > 1e-14);

	bool apply = (fit_quality_ok && sse_ok && alpha_nonzero);

#if DMFE_TAIL_FIT_DEBUG
	std::cout << "[tail_fit] best_window_global=[" << j0 << ", " << (j0 + W - 1)
			  << "] alpha_raw=" << alpha_raw
			  << " alpha_smooth=" << alpha
			  << " sse=" << best_sse
			  << " stderr=" << stderr_mean
			  << " gain=" << gain
			  << " mean_abs_res=" << mean_abs_residual
			  << " res_std=" << residual_std
			  << " apply=" << (apply ? "yes" : "no")
			  << std::endl;
#endif

	if (!apply) {
#if DMFE_TAIL_FIT_DEBUG
		if (!fit_quality_ok) {
			std::cout << "[tail_fit] skip: mean_abs_residual=" << mean_abs_residual
					  << " > 3*sigma=" << noise_threshold << std::endl;
		}
		if (!sse_ok) {
			std::cout << "[tail_fit] skip: sse=" << best_sse << " > " << sseTol << std::endl;
		}
		if (!alpha_nonzero) {
			std::cout << "[tail_fit] skip: alpha near zero" << std::endl;
		}
#endif
		return; // do not modify data
	}

	// Adaptive ramp length based on relative change in alpha
	double alpha_change = std::abs(alpha_raw - alpha) / std::max(std::abs(alpha), 1e-12);
	size_t ramp = (alpha_change < 0.005) ? 8 : 16;

	#pragma omp parallel for
	for (size_t j = j0; j < len; ++j) {
		double th = sim->h_theta[j];
		double one_m = 1.0 - th;
		double qk_fit = 1.0 - alpha * one_m * one_m;
		double w = 1.0;
		if (j < j0 + ramp) w = static_cast<double>(j - j0) / static_cast<double>(ramp);
		double qk_num = sim->h_QKv[base + j];
		sim->h_QKv[base + j] = (1.0 - w) * qk_num + w * qk_fit;
	}

	// previous alpha already updated for next iteration above
}

