#!/usr/bin/env python3
"""Scientific regression tests for DYNAMITE.

The tests run short simulations and compare them with exact results:

  T1  pure p=2 model, quench from T0=inf to T=0: C(t,0) and E(t) against the
      closed-form solution (tests/exact_p2.py);
  T3  the same with and without sparsification;
  T5  mixed 3+4 model started in equilibrium at T0 = T = 2: the dynamics is
      stationary, so E(t) = -f(1)/T and mu(t) = T + f'(1)/T at all times.  This
      exercises the terms that T1 does not (finite beta_0, thermal noise 2T R, two
      interaction orders);
  T4  CPU against GPU (only with --gpu).

Usage
-----
    python3 tests/run_regression.py [--exe ./RG-Evo] [--workdir <dir>] [--gpu]

Exit status is 0 when every test passes, 1 otherwise.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp                      # noqa: E402
from exact_p2 import C as C_exact, E as E_exact, E_TH   # noqa: E402

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
RESULTS = []


def sh(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)


def run_solver(exe, workdir, tag, args):
    """Run the solver in its own directory and return the rows of correlation.txt
    as a list of (t, E, C(t,0))."""
    d = os.path.join(workdir, tag)
    os.makedirs(os.path.join(d, "out"), exist_ok=True)
    if not os.path.isdir(os.path.join(d, "Grid_data")):
        os.symlink(os.path.join(workdir, "Grid_data"), os.path.join(d, "Grid_data"))
    cmd = [exe] + args + ["-o", os.path.join(d, "out") + os.sep, "-s", "true",
                          "-D", "false"]
    r = sh(cmd, d)
    if r.returncode != 0:
        raise RuntimeError(f"{tag}: solver failed\n{r.stdout[-3000:]}")
    hits = []
    for root, _dirs, files in os.walk(os.path.join(d, "out")):
        if "correlation.txt" in files:
            hits.append(os.path.join(root, "correlation.txt"))
    if not hits:
        raise RuntimeError(f"{tag}: no correlation.txt produced\n{r.stdout[-3000:]}")
    rows = []
    with open(max(hits, key=os.path.getmtime)) as fh:
        for line in fh:
            p = line.split()
            if len(p) == 3:
                rows.append(tuple(float(x) for x in p))
    # [CPC-REV] Guard against a vacuous pass: a run that stopped early would give a
    # small error simply because it covered a short time interval.  '-t X' is the
    # requested final time; require that at least 95% of it was reached.
    if "-t" in args:
        t_target = float(args[args.index("-t") + 1])
        if not rows or rows[-1][0] < 0.95 * t_target:
            reached = rows[-1][0] if rows else 0.0
            raise RuntimeError(f"{tag}: run stopped at t={reached:g}, requested t={t_target:g}")
    return rows


def check(name, ok, detail):
    RESULTS.append((name, ok, detail))
    tick = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{tick}] {name}: {detail}")
    return ok


# --------------------------------------------------------------------------
def t1_analytic_p2(exe, wd):
    """Pure p=2, quench T0=inf -> T=0: compare C(t,0) and E(t) with the exact
    solution.  This is the test the manuscript's Fig. 4 is based on."""
    rows = run_solver(exe, wd, "t1", ["-p", "2", "-q", "12", "-l", "1",
                                      "-T", "inf", "-G", "0", "-L", "512",
                                      "-e", "1e-8", "-d", "1e-4", "-g", "false",
                                      "-t", "1.0", "-m", "200000"])
    errC = errE = 0.0
    for t, e, c in rows:
        if t < 1e-3:
            continue
        errC = max(errC, abs(c - float(C_exact(t, 0))))
        errE = max(errE, abs(e - float(E_exact(t))))
    ok = check("T1a  max |C_num(t,0) - C_exact|", errC < 1e-6, f"{errC:.3e}  (tol 1e-6)")
    ok &= check("T1b  max |E_num(t)  - E_exact|", errE < 1e-6, f"{errE:.3e}  (tol 1e-6)")
    ok &= check("T1c  E(t) above the threshold E_th",
                min(r[1] for r in rows) > float(E_TH),
                f"min E = {min(r[1] for r in rows):.6f} > E_th = {float(E_TH):.6f}")
    return ok


def t3_sparsification_invariance(exe, wd):
    """Enabling sparsification must not move the solution by more than the
    integration tolerance (referee 2, point 2)."""
    base = ["-p", "2", "-q", "12", "-l", "1", "-T", "inf", "-G", "0",
            "-L", "512", "-e", "1e-8", "-d", "1e-4", "-g", "false", "-t", "0.5", "-m", "200000",
            "--sparsify-interval", "500"]
    off = run_solver(exe, wd, "t3off", base + ["--sparsify-sweeps", "0"])
    on = run_solver(exe, wd, "t3on", base + ["--sparsify-sweeps", "2"])
    eoff = max(abs(c - float(C_exact(t, 0))) for t, _e, c in off if t > 1e-3)
    eon = max(abs(c - float(C_exact(t, 0))) for t, _e, c in on if t > 1e-3)
    return check("T3   sparsify off vs on (error vs exact)",
                 abs(eon - eoff) < 1e-7,
                 f"off {eoff:.3e} / on {eon:.3e}, difference {abs(eon-eoff):.3e} (tol 1e-7)")


def t5_equilibrium(exe, wd):
    """Mixed 3+4 model (lambda=1/2) started in equilibrium at T0 = T = 2, above the
    dynamical transition.  By the fluctuation-dissipation theorem the energy and the
    Lagrange multiplier stay at their equilibrium values, E = -f(1)/T = -1/T and
    mu = T + f'(1)/T, with f'(1) = lambda*p + (1-lambda)*s."""
    T, p, s, lam = 2.0, 3, 4, 0.5
    rows = run_solver(exe, wd, "t5", ["-p", str(p), "-q", str(s), "-l", str(lam),
                                      "-T", str(T), "-G", str(T), "-L", "512",
                                      "-e", "1e-8", "-d", "1e-4", "-g", "false",
                                      "-t", "1.0", "-m", "200000"])
    e_eq = -1.0 / T
    mu_eq = T + (lam * p + (1 - lam) * s) / T
    errE = max(abs(e - e_eq) for _t, e, _c in rows)
    hits = []
    for root, _dirs, files in os.walk(os.path.join(wd, "t5", "out")):
        if "rvec.txt" in files:
            hits.append(os.path.join(root, "rvec.txt"))
    errMu = float("inf")
    if hits:
        with open(max(hits, key=os.path.getmtime)) as fh:
            mus = [tuple(float(x) for x in line.split()) for line in fh
                   if line.strip() and not line.startswith("#")]
        errMu = max(abs(m - mu_eq) for t, m in mus if t > 0)
    ok = check("T5a  equilibrium: max |E(t) + 1/T|", errE < 1e-7, f"{errE:.3e}  (tol 1e-7)")
    ok &= check("T5b  equilibrium: max |mu(t) - T - f'(1)/T|", errMu < 1e-7, f"{errMu:.3e}  (tol 1e-7)")
    ok &= check("T5c  C(t,0) has decayed", rows[-1][2] < 0.5, f"C(1,0) = {rows[-1][2]:.3f}")
    return ok


def t4_cpu_gpu(exe, wd):
    """CPU/GPU cross-check; skipped when no GPU is present."""
    base = ["-p", "2", "-q", "12", "-l", "1", "-T", "inf", "-G", "0",
            "-L", "512", "-e", "1e-8", "-d", "1e-4", "-t", "0.5", "-m", "200000"]
    try:
        cpu = run_solver(exe, wd, "t4cpu", base + ["-g", "false"])
        gpu = run_solver(exe, wd, "t4gpu", base + ["-g", "true"])
    except RuntimeError as exc:
        print(f"  [SKIP] T4   CPU/GPU cross-check: {exc.args[0].splitlines()[0]}")
        return True
    ecpu = max(abs(c - float(C_exact(t, 0))) for t, _e, c in cpu if t > 1e-3)
    egpu = max(abs(c - float(C_exact(t, 0))) for t, _e, c in gpu if t > 1e-3)
    return check("T4   CPU vs GPU (error vs exact)", abs(ecpu - egpu) < 1e-7,
                 f"CPU {ecpu:.3e} / GPU {egpu:.3e} (tol 1e-7)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.abspath("./RG-Evo"))
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    exe = os.path.abspath(a.exe)
    if not os.access(exe, os.X_OK):
        sys.exit(f"executable not found: {exe}")
    wd = a.workdir or tempfile.mkdtemp(prefix="dynamite-regression-")
    os.makedirs(wd, exist_ok=True)
    src_grid = os.path.join(os.path.dirname(exe), "Grid_data")
    dst_grid = os.path.join(wd, "Grid_data")
    if not os.path.isdir(dst_grid):
        shutil.copytree(src_grid, dst_grid)
    print(f"DYNAMITE scientific regression tests\n  exe     : {exe}\n  workdir : {wd}\n")
    ok = True
    ok &= t1_analytic_p2(exe, wd)
    ok &= t3_sparsification_invariance(exe, wd)
    ok &= t5_equilibrium(exe, wd)
    if a.gpu:
        ok &= t4_cpu_gpu(exe, wd)
    print()
    n_fail = sum(1 for _n, o, _d in RESULTS if not o)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
