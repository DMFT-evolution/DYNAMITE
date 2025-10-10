#pragma once

#include <boost/multiprecision/cpp_dec_float.hpp>

namespace dmfe {

// High-precision decimal type used across DMFE for exact grid parameterization.
using mp = boost::multiprecision::cpp_dec_float_100;

// Lambert W function, branch -1, for x in [-1/e, 0).
// Throws std::domain_error if x is outside the branch domain.
// Implemented with Halley's method with robust initial guesses.
mp lambertWm1_mp(const mp& x);

} // namespace dmfe
