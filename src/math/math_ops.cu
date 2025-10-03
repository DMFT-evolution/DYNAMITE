#include "math/math_ops.hpp"
#include "core/config.hpp"            // SimulationConfig values (runtime)
#include "core/device_constants.hpp"   // device __constant__: d_lambda, d_p, d_p2

// GPU-optimized device function for integer power computation
__device__ __forceinline__ double fast_pow_int(double base, int exp) {
	if (exp == 0) return 1.0;
	if (exp == 1) return base;
	if (exp == 2) return base * base;
	if (exp == 3) return base * base * base;
	if (exp == 4) { double sq = base * base; return sq * sq; }
	if (exp == 5) { double sq = base * base; return sq * sq * base; }
	if (exp == 6) { double sq = base * base; return sq * sq * sq; }
	if (exp == 7) { double sq = base * base; double cu = sq * base; return cu * cu * base; }
	if (exp == 8) { double sq = base * base; double qu = sq * sq; return qu * qu; }
	if (exp == 9) { double cu = base * base * base; return cu * cu * cu; }
	if (exp == 10) { double sq = base * base; double qu = sq * sq; return qu * qu * sq; }
	if (exp == 11) { double sq = base * base; double qu = sq * sq; return qu * qu * sq * base; }
	if (exp == 12) { double cu = base * base * base; double si = cu * cu; return si * si; }
	if (exp == 13) { double cu = base * base * base; double si = cu * cu; return si * si * base; }
	if (exp == 14) { double sq = base * base; double qu = sq * sq; double oc = qu * qu; return oc * oc * sq; }
	double result = 1.0;
	double current = base;
	while (exp > 0) {
		if (exp & 1) result *= current;
		current *= current;
		exp >>= 1;
	}
	return result;
}

// GPU device functions using device constants
__device__ double flambdaGPU(double q) { return d_lambda * fast_pow_int(q, d_p) + (1 - d_lambda) * fast_pow_int(q, d_p2); }
__device__ double DflambdaGPU(double q) { return d_lambda * d_p * fast_pow_int(q, d_p - 1) + (1 - d_lambda) * d_p2 * fast_pow_int(q, d_p2 - 1); }
__device__ double DflambdaGPU2(double q) {
	double term1 = 0.0, term2 = 0.0;
	if (d_p - 2 >= 0) term1 = d_lambda * d_p * (d_p - 1) * fast_pow_int(q, d_p - 2);
	if (d_p2 - 2 >= 0) term2 = (1 - d_lambda) * d_p2 * (d_p2 - 1) * fast_pow_int(q, d_p2 - 2);
	return term1 + term2;
}
__device__ double DDflambdaGPU(double q) { return d_lambda * d_p * (d_p - 1) * fast_pow_int(q, d_p - 2) + (1 - d_lambda) * d_p2 * (d_p2 - 1) * fast_pow_int(q, d_p2 - 2); }
__device__ double DDDflambdaGPU(double q) { return d_lambda * d_p * (d_p - 1) * (d_p - 2) * fast_pow_int(q, d_p - 3) + (1 - d_lambda) * d_p2 * (d_p2 - 1) * (d_p2 - 2) * fast_pow_int(q, d_p2 - 3); }

