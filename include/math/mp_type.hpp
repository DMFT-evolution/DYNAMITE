#pragma once

// Lightweight abstraction over high-precision decimal type.
// If Boost.Multiprecision is available, use cpp_dec_float_100; else fall back to long double.

#if __has_include(<boost/multiprecision/cpp_dec_float.hpp>)
  #define DMFE_HAVE_BOOST_MP 1
  #include <boost/multiprecision/cpp_dec_float.hpp>
  namespace dmfe {
  using mp = boost::multiprecision::cpp_dec_float_100;
  }
#else
  #define DMFE_HAVE_BOOST_MP 0
  #include <cmath>
  namespace dmfe {
  using mp = long double;
  }
#endif

namespace dmfe {
// Pi helper usable with either backend
inline mp mp_pi() {
#if DMFE_HAVE_BOOST_MP
  return boost::math::constants::pi<mp>();
#else
  return static_cast<mp>(3.141592653589793238462643383279502884L);
#endif
}
}
