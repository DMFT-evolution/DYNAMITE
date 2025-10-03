#include "math/math_ops.hpp"
#include "core/config.hpp"

// Access runtime-configured parameters via global config
extern SimulationConfig config;

double pow_int(double base, int exp) {
	if (exp == 0) return 1.0;
	if (exp == 1) return base;
	if (exp == 2) return base * base;
	if (exp == 3) return base * base * base;
	if (exp == 4) { double sq = base * base; return sq * sq; }
	double result = 1.0;
	double current = base;
	while (exp > 0) {
		if (exp & 1) result *= current;
		current *= current;
		exp >>= 1;
	}
	return result;
}

double flambda(double q) { 
	return config.lambda * pow_int(q, config.p) + (1 - config.lambda) * pow_int(q, config.p2); 
}

double Dflambda(double q) { 
	return config.lambda * config.p * pow_int(q, config.p - 1) + (1 - config.lambda) * config.p2 * pow_int(q, config.p2 - 1); 
}

double DDflambda(double q) { 
	return config.lambda * config.p * (config.p - 1) * pow_int(q, config.p - 2) + (1 - config.lambda) * config.p2 * (config.p2 - 1) * pow_int(q, config.p2 - 2); 
}

double DDDflambda(double q) { 
	return config.lambda * config.p * (config.p - 1) * (config.p - 2) * pow_int(q, config.p - 3) + (1 - config.lambda) * config.p2 * (config.p2 - 1) * (config.p2 - 2) * pow_int(q, config.p2 - 3); 
}
