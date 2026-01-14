#!/usr/bin/env python3
r"""Plot the threshold-energy gap: E(t) - E_th on log-log axes.

DYNAMITE writes `energy.txt` and `params.txt` into each output directory.
This script:
    1) reads E(t) from energy.txt (2+ columns: time, energy)
    2) reads (lambda, p, q) from params.txt
    3) computes the threshold energy E_th from

    f_λ(x) = λ x^p + (1-λ) x^q

For $\Gamma = 0$ ("zero final temperature"), we use the simple closed form:

    E_th = sqrt(f''_λ(1)) * ( f_λ(1)/f''_λ(1) - f'_λ(1)/f''_λ(1) - f_λ(1)/f'_λ(1) )

For $\Gamma \neq 0$ (finite final temperature), we first solve for $q_1 \in (0,1)$:

    1/(1-q_1)^2 - f''_λ(q_1)/Gamma^2 = 0

and then compute:

    E_th = (-f_λ(1) + f_λ(q1) * ( 1/q1 + Gamma^2/((q1-1) f'_λ(q1)) )) / Gamma

Notes:
- In the C++ codebase the second exponent is typically called p2; this script
  accepts either `q` or `p2` in params.txt.
- Lines starting with '#' in energy.txt are treated as comments.

Usage:
    python3 scripts/plot_energy_threshold_gap.py /path/to/output/dir

Outputs:
  - shows an interactive window (default)
  - or saves to a file via --out
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def load_two_column_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # Prefer numpy parsing, but fall back to a small manual parser that tolerates
    # both real tabs/spaces and the literal two-character sequence "\\t".
    try:
        data = np.loadtxt(path, comments="#")
    except ValueError:
        rows: list[tuple[float, float]] = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # Some users paste examples with a literal "\t" in them.
                line = line.replace("\\t", " ")
                parts = line.split()
                if len(parts) < 2:
                    continue
                rows.append((float(parts[0]), float(parts[1])))
        if not rows:
            raise
        data = np.array(rows, dtype=float)

    if data.ndim == 1:
        if data.size < 2:
            raise ValueError(f"Expected at least 2 columns in {path}")
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}; got {data.shape[1]}")
    return data[:, 0], data[:, 1]


def parse_params_txt(path: Path) -> dict[str, str]:
    """Parse simple `key = value` lines; ignores unparseable lines."""
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def get_float(params: dict[str, str], key: str) -> float:
    if key not in params:
        raise KeyError(key)
    raw = params[key].strip()
    # Allow common spellings used in params files.
    if raw.lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if raw.lower() in {"-inf", "-infinity"}:
        return float("-inf")
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Could not parse {key}={params[key]!r} as float") from e


def get_int(params: dict[str, str], key: str) -> int:
    if key not in params:
        raise KeyError(key)
    try:
        return int(float(params[key]))
    except ValueError as e:
        raise ValueError(f"Could not parse {key}={params[key]!r} as int") from e


def f_lambda(x: float, lam: float, p: int, q: int) -> float:
    return lam * (x**p) + (1.0 - lam) * (x**q)


def df_lambda(x: float, lam: float, p: int, q: int) -> float:
    return lam * p * (x ** (p - 1)) + (1.0 - lam) * q * (x ** (q - 1))


def ddf_lambda(x: float, lam: float, p: int, q: int) -> float:
    return lam * p * (p - 1) * (x ** (p - 2)) + (1.0 - lam) * q * (q - 1) * (x ** (q - 2))


def solve_q1(gamma: float, lam: float, p: int, q: int) -> float:
    """Solve for q1 in (0,1) using a simple bracket+bisection.

    Equation:
        1/(1-q)^2 - f''(q)/gamma^2 = 0
    """

    if not (gamma > 0):
        return 1.0

    def gfun(x: float) -> float:
        return 1.0 / ((1.0 - x) ** 2) - ddf_lambda(x, lam, p, q) / (gamma**2)

    # Search for sign changes from low -> high.
    # We want the *largest* solution in (0,1), matching the Mathematica
    # convention of selecting the last solution returned by NSolve.
    lo = 1e-12
    hi = 1.0 - 1e-10

    # Coarse scan (geometric grid) to bracket roots.
    prev_x = lo
    prev_g = gfun(prev_x)
    brackets: list[tuple[float, float]] = []
    for x in np.geomspace(1e-10, 1.0 - 1e-8, 320):
        if x >= hi:
            break
        gx = gfun(float(x))
        if math.isfinite(prev_g) and prev_g == 0.0:
            brackets.append((float(prev_x), float(prev_x)))
        elif math.isfinite(prev_g) and math.isfinite(gx) and (prev_g * gx) < 0:
            brackets.append((float(prev_x), float(x)))
        prev_x, prev_g = float(x), gx

    if not brackets:
        # As a last attempt, try the full interval if it's well-defined.
        g_lo = gfun(lo)
        g_hi = gfun(hi)
        if math.isfinite(g_lo) and math.isfinite(g_hi) and (g_lo * g_hi) < 0:
            brackets.append((lo, hi))

    if not brackets:
        raise ValueError(
            "Could not bracket a root for q1 in (0,1). "
            "This can happen for unexpected parameter regimes."
        )

    # Pick the largest-root bracket (closest to 1).
    a, b = max(brackets, key=lambda ab: ab[1])
    fa = gfun(a)
    fb = gfun(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        raise ValueError("Internal error: invalid bracket for q1")

    # Bisection
    for _ in range(120):
        m = 0.5 * (a + b)
        fm = gfun(m)
        if not math.isfinite(fm):
            # Nudge away from singularities
            m = 0.5 * (m + a)
            fm = gfun(m)
        if fm == 0.0:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        if abs(b - a) / max(1.0, abs(m)) < 1e-13:
            break
    return 0.5 * (a + b)


def compute_eth(lam: float, p: int, q: int, gamma: float) -> tuple[float, float]:
    """Return (Eth, q1). Uses gamma-dependent definition when gamma!=0."""

    # Treat +inf like gamma != 0 (the formula still works due to 1/gamma prefactor
    # but q1 equation becomes stiff; in practice DMFE uses Gamma=0 for the
    # "zero-temperature" expression).
    if gamma == 0.0:
        f1 = f_lambda(1.0, lam, p, q)
        fp1 = df_lambda(1.0, lam, p, q)
        fpp1 = ddf_lambda(1.0, lam, p, q)

        if fpp1 <= 0:
            raise ValueError(f"f''_lambda(1) must be > 0 to define Eth; got {fpp1}")
        if fp1 == 0:
            raise ValueError("f'_lambda(1) is zero; Eth formula divides by f'(1)")

        eth0 = math.sqrt(fpp1) * (f1 / fpp1 - fp1 / fpp1 - f1 / fp1)
        return float(eth0), 1.0

    if not math.isfinite(gamma):
        raise ValueError(f"Gamma must be finite for the finite-temperature Eth formula; got {gamma}")
    if gamma < 0:
        raise ValueError(f"Gamma must be >= 0; got {gamma}")

    q1 = solve_q1(gamma, lam, p, q)
    fq1 = f_lambda(q1, lam, p, q)
    fpq1 = df_lambda(q1, lam, p, q)
    if fpq1 == 0:
        raise ValueError("f'_lambda(q1) is zero; finite-temperature Eth formula divides by it")

    eth = (-f_lambda(1.0, lam, p, q) + fq1 * (1.0 / q1 + (gamma**2) / ((q1 - 1.0) * fpq1))) / gamma
    return float(eth), float(q1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Log-log plot of E(t) - Eth from a DYNAMITE output directory"
    )
    ap.add_argument("output_dir", type=Path, help="Run output directory containing energy.txt and params.txt")
    ap.add_argument("--out", type=Path, default=None, help="Write figure to this file (e.g. eth_gap.png)")
    ap.add_argument(
        "--skip-nonpositive",
        action="store_true",
        help="Drop points where t<=0 or (E-Eth)<=0 before log-log plotting",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=None,
        help="If set, clamp (Eth-E) to at least eps to allow log plotting",
    )
    ap.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title (default: auto from parameters)",
    )
    args = ap.parse_args()

    energy_path = args.output_dir / "energy.txt"
    params_path = args.output_dir / "params.txt"
    if not energy_path.exists():
        raise FileNotFoundError(f"Missing {energy_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"Missing {params_path}")

    t, e = load_two_column_txt(energy_path)

    params = parse_params_txt(params_path)
    lam = get_float(params, "lambda")
    p = get_int(params, "p")
    # accept either q or p2 (repo convention)
    if "q" in params:
        q = get_int(params, "q")
    elif "p2" in params:
        q = get_int(params, "p2")
    else:
        raise KeyError("Expected 'q' or 'p2' in params.txt")

    gamma = get_float(params, "Gamma")
    eth, q1 = compute_eth(lam, p, q, gamma)
    gap = e - eth

    # Always drop nonpositive times: log-log plots require t > 0.
    # (This is independent of whether we drop nonpositive gaps.)
    t_pos = t > 0
    t = t[t_pos]
    gap = gap[t_pos]

    if args.eps is not None:
        gap = np.maximum(gap, float(args.eps))

    if args.skip_nonpositive:
        mask = gap > 0
        t = t[mask]
        gap = gap[mask]

    if t.size == 0:
        raise ValueError(
            "No data points left to plot after filtering. "
            "This usually means E(t) - Eth <= 0 for all saved times (e.g. Eth computed above the trajectory energy). "
            "Check lambda/p/q (q may be stored as p2), or use --eps to clamp the gap for visualization."
        )

    if not np.all(t > 0) or not np.all(gap > 0):
        raise ValueError(
            "Log-log requires t>0 and (E-Eth)>0 for all points. "
            "Use --skip-nonpositive and/or --eps to handle early/negative points."
        )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=130)
    ax.plot(t, gap, lw=1.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$E(t) - E_{\mathrm{th}}$")
    ax.grid(True, which="both", alpha=0.25)

    if args.title is not None:
        ax.set_title(args.title)
    else:
        if gamma == 0.0:
            ax.set_title(f"$\\lambda$={lam:g}, p={p}, q={q}, $\\Gamma$=0, $E_{{th}}$={eth:.6g}")
        else:
            ax.set_title(f"$\\lambda$={lam:g}, p={p}, q={q}, $\\Gamma$={gamma:g}, $E_{{th}}$={eth:.6g}")

    fig.tight_layout()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out)
        print(f"Wrote {args.out}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
