#include "math/lambert_w.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace dmfe {

mp lambertWm1_mp(const mp& x) {
    if (!(x <= mp(0))) throw std::domain_error("lambertWm1 domain x<=0");
    const mp minus_inv_e = -mp(1) / exp(mp(1));
    if (x < minus_inv_e) throw std::domain_error("lambertWm1 domain x>=-1/e");
    if (x == mp(0)) return -std::numeric_limits<double>::infinity();

    mp w;
    if (x > mp(-0.1)) {
        mp L1 = log(-x);
        mp L2 = log(-L1);
        w = L1 - L2 + (L2 / L1);
    } else {
        mp u = x * exp(mp(1)) + mp(1);
        if (u > mp(0)) u = mp(0);
        if (u < mp(-1)) u = mp(-1);
        mp s = sqrt(-mp(2) * u);
        w = -mp(1) - s + (s*s)/mp(3);
    }

    for (int iter = 0; iter < 80; ++iter) {
        mp ew = exp(w);
        mp wew = w * ew;
        mp f = wew - x;
        mp wp1 = w + mp(1);
        mp denom = ew * wp1 - (wp1 + mp(1)) * f / (mp(2) * wp1);
        mp dw = f / denom;
        mp wnext = w - dw;
        if (!std::isfinite(static_cast<double>(wnext))) {
            mp fprime = ew * (w + mp(1));
            dw = f / fprime;
            wnext = w - dw;
        }
        w = wnext;
        if (abs(dw) <= std::numeric_limits<double>::epsilon()) break;
    }
    return w;
}

} // namespace dmfe
