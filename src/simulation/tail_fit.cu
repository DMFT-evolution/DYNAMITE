#include "simulation/tail_fit.hpp"
#include "simulation/simulation_data.hpp"
#include "core/config.hpp"
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <limits>
#include <iostream>
#include <cmath>
#include <algorithm>

#if DMFE_WITH_CUDA
extern SimulationConfig config;
extern SimulationData* sim;
extern int last_rollback_loop; // Track rollbacks to reset smoothing

// Persistent state across calls (matches CPU implementation)
static size_t g_prev_best_off = static_cast<size_t>(-1);
static size_t g_prev_t1len = 0;
static double g_alpha_smoothed = 0.0;
static int g_last_known_rollback = -1000; // Track last rollback we've seen

// Optimized kernel: compute y array for tail region
// Uses vectorized loads and coalesced memory access
__global__ void k_compute_y(const double* __restrict__ theta,
							const double* __restrict__ QK,
							double* __restrict__ y,
							size_t len, size_t base, size_t start) {
	size_t j = blockIdx.x * blockDim.x + threadIdx.x + start;
	if (j < len) {
		double th = theta[j];
		double one_m = 1.0 - th;
		double denom = one_m * one_m;
		double qk = QK[base + j];
		double val = (1.0 - qk) / fmax(denom, 1e-300);
		y[j - start] = val;
	}
}

// Optimized kernel: compute window statistics (mean and SSE) in parallel
// Each block computes one window using shared memory reduction
__global__ void k_window_stats(const double* __restrict__ y,
							   size_t N, size_t W,
							   const size_t* __restrict__ offsets,
							   size_t num_windows,
							   double* __restrict__ means,
							   double* __restrict__ sses) {
	__shared__ double s_sum[256];
	__shared__ double s_sumsq[256];
	
	size_t wid = blockIdx.x;
	if (wid >= num_windows) return;
	
	size_t off = offsets[wid];
	size_t tid = threadIdx.x;
	size_t block_size = blockDim.x;
	
	// Each thread accumulates its portion
	double sum = 0.0;
	double sumsq = 0.0;
	for (size_t k = tid; k < W; k += block_size) {
		if (off + k < N) {
			double val = y[off + k];
			sum += val;
			sumsq += val * val;
		}
	}
	
	s_sum[tid] = sum;
	s_sumsq[tid] = sumsq;
	__syncthreads();
	
	// Block reduction
	for (size_t s = block_size / 2; s > 0; s >>= 1) {
		if (tid < s) {
			s_sum[tid] += s_sum[tid + s];
			s_sumsq[tid] += s_sumsq[tid + s];
		}
		__syncthreads();
	}
	
	if (tid == 0) {
		double mean = s_sum[0] / static_cast<double>(W);
		double sse = s_sumsq[0] - 2.0 * mean * s_sum[0] + static_cast<double>(W) * mean * mean;
		means[wid] = mean;
		sses[wid] = sse;
	}
}

// Optimized kernel: compute residuals for tail region
__global__ void k_compute_residuals(const double* __restrict__ theta,
									const double* __restrict__ QK,
									double* __restrict__ residuals,
									size_t len, size_t base,
									size_t j0, double alpha) {
	size_t j = blockIdx.x * blockDim.x + threadIdx.x + j0;
	if (j < len) {
		double th = theta[j];
		double one_m = 1.0 - th;
		double qk_fit = 1.0 - alpha * one_m * one_m;
		double qk_num = QK[base + j];
		residuals[j - j0] = qk_fit - qk_num;
	}
}

// Optimized kernel: blend with adaptive ramp
__global__ void k_blend_qk(double* __restrict__ QK,
						   const double* __restrict__ theta,
						   size_t len, size_t base,
						   size_t j0, double alpha, size_t ramp) {
	size_t j = blockIdx.x * blockDim.x + threadIdx.x + j0;
	if (j < len) {
		double th = theta[j];
		double one_m = 1.0 - th;
		double qk_fit = 1.0 - alpha * one_m * one_m;
		double w = (j < j0 + ramp) ? static_cast<double>(j - j0) / static_cast<double>(ramp) : 1.0;
		double qk_num = QK[base + j];
		QK[base + j] = (1.0 - w) * qk_num + w * qk_fit;
	}
}

void tailFitBlendGPU() {
	const size_t len = config.len;
	if (len < 32) return;
	const size_t t1len = sim->d_t1grid.size();
	if (t1len == 0) return;
	const size_t base = (t1len - 1) * len;
	const size_t start = len / 2;
	const size_t W = 16; // Match CPU implementation
	if (len - start < W) return;

	// Check if this is a new time step
	const bool new_time_step = (g_prev_t1len != t1len);
	
	// Compute y values for tail region
	const size_t N = len - start;
	sim->temp10.resize(N);
	double* yptr = thrust::raw_pointer_cast(sim->temp10.data());
	const double* thptr = thrust::raw_pointer_cast(sim->d_theta.data());
	double* qkptr = thrust::raw_pointer_cast(sim->d_QKv.data());
	
	dim3 block(256);
	dim3 grid((N + block.x - 1) / block.x);
	k_compute_y<<<grid, block>>>(thptr, qkptr, yptr, len, base, start);
	cudaDeviceSynchronize(); // Ensure y values are computed before proceeding

	// Determine candidate windows
	std::vector<size_t> h_candidates;
	if (g_prev_best_off == static_cast<size_t>(-1) || !new_time_step) {
		// Full scan: every W-th offset
		size_t window_count = (N - W) / W + 1;
		for (size_t w = 0; w < window_count; ++w) {
			h_candidates.push_back(w * W);
		}
	} else {
		// Restricted scan: previous best and neighbors
		h_candidates.push_back(g_prev_best_off);
		if (g_prev_best_off >= W) h_candidates.push_back(g_prev_best_off - W);
		if (g_prev_best_off + W + W <= N) h_candidates.push_back(g_prev_best_off + W);
		std::sort(h_candidates.begin(), h_candidates.end());
		h_candidates.erase(std::unique(h_candidates.begin(), h_candidates.end()), h_candidates.end());
	}
	
	// Filter valid candidates
	std::vector<size_t> valid_candidates;
	for (size_t off : h_candidates) {
		if (off + W <= N) valid_candidates.push_back(off);
	}
	
	if (valid_candidates.empty()) return;
	
	// Upload candidates and allocate results
	const size_t num_windows = valid_candidates.size();
	sim->Stemp2.resize(num_windows);
	sim->temp11.resize(num_windows);
	sim->temp10.resize(num_windows);
	
	thrust::copy(valid_candidates.begin(), valid_candidates.end(), sim->Stemp2.begin());
	const size_t* d_offsets = thrust::raw_pointer_cast(sim->Stemp2.data());
	double* d_means = thrust::raw_pointer_cast(sim->temp11.data());
	double* d_sses = thrust::raw_pointer_cast(sim->temp10.data());
	
	// Compute window statistics on GPU
	dim3 stat_block(256);
	dim3 stat_grid(num_windows);
	k_window_stats<<<stat_grid, stat_block>>>(yptr, N, W, d_offsets, num_windows, d_means, d_sses);
	cudaDeviceSynchronize(); // Ensure window stats are computed before copying back
	
	// Copy results to host and find best
	std::vector<double> h_means(num_windows), h_sses(num_windows);
	thrust::copy(sim->temp11.begin(), sim->temp11.begin() + num_windows, h_means.begin());
	thrust::copy(sim->temp10.begin(), sim->temp10.begin() + num_windows, h_sses.begin());
	
	size_t best_idx = 0;
	double best_sse = h_sses[0];
	for (size_t i = 1; i < num_windows; ++i) {
		if (h_sses[i] < best_sse) {
			best_sse = h_sses[i];
			best_idx = i;
		}
	}
	
	const size_t best_off = valid_candidates[best_idx];
	const double alpha_raw = h_means[best_idx];
	const size_t j0 = start + best_off;
	
	// Update persistent state
	g_prev_best_off = best_off;
	g_prev_t1len = t1len;
	
	// Estimate variance and standard error
	const double var = (W > 1) ? (best_sse / static_cast<double>(W - 1)) : 0.0;
	const double stderr_mean = (W > 0) ? std::sqrt(std::max(var, 0.0) / static_cast<double>(W)) : 0.0;
	
	// Detect rollback: if last_rollback_loop changed, reset smoothing to raw value
	bool rollback_detected = (last_rollback_loop != g_last_known_rollback);
	if (rollback_detected) {
		g_last_known_rollback = last_rollback_loop;
		if (alpha_raw != 0.0) {
			g_alpha_smoothed = alpha_raw;
#if DMFE_TAIL_FIT_DEBUG
			if (config.debug) {
				std::cout << "[tail_fit] Rollback detected (loop=" << last_rollback_loop 
						  << "), resetting alpha_smooth=" << alpha_raw << std::endl;
			}
#endif
		}
	}
	
	// Adaptive smoothing gain - increased smoothing (lower max_gain, stronger damping)
	const double max_gain = 0.50; 
	const double min_gain = 0.10;
	double rel_err = (std::abs(alpha_raw) > 1e-12) ? stderr_mean / std::abs(alpha_raw) : 1.0;
	double gain = max_gain / (1.0 + 2.0 * rel_err);
	gain = std::max(min_gain, std::min(gain, max_gain));
	
	// Initialize or update smoothed alpha
	if (!(g_alpha_smoothed == g_alpha_smoothed) || (alpha_raw == 0.0 && rollback_detected)) {
		// Initialize if NaN or if rollback with zero raw value
		if (alpha_raw != 0.0) {
			g_alpha_smoothed = alpha_raw;
		} else if (!(g_alpha_smoothed == g_alpha_smoothed)) {
			g_alpha_smoothed = 0.0;
		}
		// else: keep existing alpha_smoothed if raw is zero but we have a valid smoothed value
	} else if (alpha_raw == 0.0) {
		// If raw is zero but smoothed is valid, keep smoothed (don't update)
		// This allows continuation with previous valid smoothed value
	} else {
		// Normal case: apply exponential moving average
		g_alpha_smoothed = (1.0 - gain) * g_alpha_smoothed + gain * alpha_raw;
	}
	const double alpha = g_alpha_smoothed;
	
	// Compute residuals on GPU
	const size_t tail_len = len - j0;
	sim->temp12.resize(tail_len);
	double* d_residuals = thrust::raw_pointer_cast(sim->temp12.data());
	
	dim3 res_grid((tail_len + block.x - 1) / block.x);
	k_compute_residuals<<<res_grid, block>>>(thptr, qkptr, d_residuals, len, base, j0, alpha);
	cudaDeviceSynchronize(); // Ensure residuals are computed before thrust operations
	
	// Compute residual statistics using Thrust
	double mean_abs_residual = thrust::transform_reduce(
		thrust::device,
		sim->temp12.begin(), sim->temp12.begin() + tail_len,
		[] __host__ __device__ (double r) -> double { return fabs(r); },
		0.0,
		thrust::plus<double>()
	) / static_cast<double>(tail_len);
	
	double mean_residual = thrust::reduce(
		thrust::device,
		sim->temp12.begin(), sim->temp12.begin() + tail_len,
		0.0,
		thrust::plus<double>()
	) / static_cast<double>(tail_len);
	
	double residual_variance = thrust::transform_reduce(
		thrust::device,
		sim->temp12.begin(), sim->temp12.begin() + tail_len,
		[mean_residual] __host__ __device__ (double r) -> double {
			double d = r - mean_residual;
			return d * d;
		},
		0.0,
		thrust::plus<double>()
	) / std::max(1.0, static_cast<double>(tail_len) - 1.0);
	
	const double residual_std = std::sqrt(std::max(0.0, residual_variance));
	
	// Decision gates matching CPU implementation
	const double noise_threshold = 3.0 * residual_std;
	const bool fit_quality_ok = (mean_abs_residual <= noise_threshold);
	const double sseTol = std::sqrt(config.delta_max);
	const bool sse_ok = (best_sse <= sseTol);
	const bool alpha_nonzero = (std::abs(alpha) > 1e-14);
	const bool apply = (fit_quality_ok && sse_ok && alpha_nonzero);
	
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
		return;
	}
	
	// Adaptive ramp length
	double alpha_change = std::abs(alpha_raw - alpha) / std::max(std::abs(alpha), 1e-12);
	size_t ramp = (alpha_change < 0.005) ? 8 : 16;
	
	// Apply blending on GPU
	dim3 blend_grid((tail_len + block.x - 1) / block.x);
	k_blend_qk<<<blend_grid, block>>>(qkptr, thptr, len, base, j0, alpha, ramp);
	cudaDeviceSynchronize();
}
#endif

