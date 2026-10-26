#!/usr/bin/env python3
"""[CPC-REV] Exact solution of the PURE spherical p=2 model after a quench from
T0 = infinity to T = 0, used as the analytic benchmark of the regression suite.

Convention: f(q) = lambda q^p + (1-lambda) q^s  (the one implemented in
src/math/math_ops.cpp).  For the pure p = 2 model, f(q) = q^2, hence
f'(1) = 2 and J := sqrt(f''(1)) = sqrt(2).

With b(tau) = I_1(2 J tau) / (J tau) and g(t) = sqrt(b(2t)):

    C(t,t') = b(t+t') / ( g(t) g(t') )
    R(t,t') = [ g(t') / g(t) ] b(t-t')            (t > t')
    mu(t)   = d/dt log g(t) = 2 J I_0(4Jt)/I_1(4Jt) - 1/t
    E(t)    = -mu(t)/2 = -J I_0(4Jt)/I_1(4Jt) + 1/(2t)
    E_th    = -sqrt(2)  ( = -2 sqrt((p-1)/p) for p = 2 )

The residuals of Eqs. (1) and (13) of the manuscript vanish to 30+ digits for
these expressions (see tests/test_exact_p2_selfcheck.py).
"""
import mpmath as mp

mp.mp.dps = 40
J = mp.sqrt(2)


def b(tau):
    tau = mp.mpf(tau)
    if tau == 0:
        return mp.mpf(1)
    return mp.besseli(1, 2 * J * tau) / (J * tau)


def g(t):
    return mp.sqrt(b(2 * mp.mpf(t)))


def C(t, tp):
    return b(mp.mpf(t) + mp.mpf(tp)) / (g(t) * g(tp))


def R(t, tp):
    return g(tp) / g(t) * b(mp.mpf(t) - mp.mpf(tp))


def mu(t):
    t = mp.mpf(t)
    x = 4 * J * t
    return 2 * J * mp.besseli(0, x) / mp.besseli(1, x) - 1 / t


def E(t):
    return -mu(t) / 2


E_TH = -mp.sqrt(2)

if __name__ == "__main__":
    print("t            C(t,0)                 E(t)")
    for t in ("0.01", "0.1", "1", "10", "100"):
        print(f"{t:<12} {mp.nstr(C(t,0),16):<22} {mp.nstr(E(t),16)}")
    print("E_th =", mp.nstr(E_TH, 16))
