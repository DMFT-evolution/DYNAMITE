#pragma once

#include "math/mp_type.hpp"

namespace dmfe {

// High-precision decimal type (Boost or long double fallback)
using mp = dmfe::mp;

// Lambert W function, branch -1, for x in [-1/e, 0).
// Throws std::domain_error if x is outside the branch domain.
// Implemented with Halley's method with robust initial guesses.
mp lambertWm1_mp(const mp& x);

} // namespace dmfe
