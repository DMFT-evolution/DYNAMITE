# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/algorithm.svg" alt="Usage icon"/> Usage

Entry point: `./RG-Evo`. Show all options with `-h`.

## Model and parameters

- `-p`, `-q` (integers): model orders (e.g., spherical mixed p-spin). Use `p>=2`; common: `p=3, q=4`.
- `-l, --lambda` (float): coupling strength (dimensionless). Typical 0–1; study transitions by sweeping.
- `-T, --T0` (float|inf): initial temperature; use `inf` for zero-noise quenches.
- `-G, --Gamma` (float): final temperature.

Note: At present, the mixed spherical p-spin equations are hardcoded as in the paper; other models are not yet configurable. This will be relaxed in a future release.

## Discretization and accuracy

- `-L` grid length: selects `Grid_data/<L>/` (available: 512/1024/2048). Larger L → higher accuracy and cost.
- `-m` max iterations and `-t` max physical time bound the run. For exploratory runs, start with `-m 1e4`.
- `-d` minimum time step and `-e` error tolerance control adaptivity (RK54 default with auto-switch to SSPRK104; SERK2 trials optional). Smaller `-e` → better accuracy.

## Execution controls

- `-g, --gpu` boolean: enable GPU kernels (default true when available); set `false` for CPU-only or reproducibility.
- `-A, --async-export` boolean: asynchronous I/O to avoid blocking the integrator (default true).
- `-s` save outputs (default true) and `-o` output directory root.
- `-D` debug logging; `-v` print build/version; `-I` allow resume across incompatible versions (use with care).

Interpolation mode:
- `-R, --log-response-interp` boolean: interpolate QR and dQR in log space when safe (default false). If enabled, the code interpolates f=log(QR) and g=dQR/QR and exponentiates back; it automatically falls back to linear when any stencil QR<=0. QK/dQK remain linear.

Sparsification:
- `-w, --sparsify-sweeps INT`: number of sparsify sweeps per maintenance pass.
	- `-1` (default): auto — CPU: 1 sweep; GPU: 1 sweep normally, 2 sweeps if GPU memory usage > 50%.
	- `0`: disable sparsification.
	- `>0`: fixed number of sweeps.

## Inputs (grids)

Interpolation weights/indices are loaded from `Grid_data/<L>/`:
- `theta.dat`, `phi1.dat`, `phi2.dat`, `int.dat`
- `posA1y.dat`, `posA2y.dat`, `posB2y.dat`
- `indsA1y.dat`, `indsA2y.dat`, `indsB2y.dat`
- `weightsA1y.dat`, `weightsA2y.dat`, `weightsB2y.dat`

Choose the largest L that fits memory/time for your study; verify convergence of observables with L.

### Generating grids and interpolation metadata

Use the built-in grid generator to create or refresh `Grid_data/<L>/`:

```bash
./RG-Evo grid [--len L] [--Tmax X] [--dir SUBDIR] \
							[--alpha X] [--delta X] \
							[--spline-order n] [--interp-method METHOD] [--interp-order n]
# METHODS: poly | rational | bspline
```

This writes:
- `theta.dat` (N), `phi1.dat` (N×N), `phi2.dat` (N×N)
- `int.dat` (N) — integration weights (controlled by `--spline-order`)
- `posA1y.dat`, `posA2y.dat`, `posB2y.dat` (each N×N)
- Interpolation metadata for mapping theta → targets:
	- A1 (phi1): `indsA1y.dat`, `weightsA1y.dat`
	- A2 (phi2): `indsA2y.dat`, `weightsA2y.dat`
	- B2 (theta/(phi2−1e−200)): `indsB2y.dat`, `weightsB2y.dat`

Alpha/Delta (optional):
- `--alpha` in [0,1] blends a smooth non-linear index remapping into the default mapping; `--delta >= 0` controls the softness. Defaults are 0, preserving the paper grid exactly. The chosen values are recorded in `grid_params.txt`.

Method notes and trade‑offs:
- `poly`: local barycentric Lagrange of order n (degree n). Minimal weights per entry (n+1). Good accuracy for smooth data; fastest when the sample values change often.
- `rational`: barycentric rational variant with the same interface and locality as `poly`, typically slightly more robust on irregular grids.
- `bspline`: B‑spline of degree n via global collocation. Exports dense weights per entry (global linear map). Prefer this when you need spline smoothness/derivatives and can reuse the same data vector across many evaluations; otherwise `poly`/`rational` are usually faster.

## Outputs and observables

- HDF5 `data.h5` when available; else `data.bin` plus text summaries.
- Datasets: `QKv`, `QRv`, `dQKv`, `dQRv`, `t1grid`, `rvec`, `drvec` (see concepts/eoms-and-observables.md).
- Text: `params.txt`, `correlation.txt`, `energy.txt`, `rvec.txt`, `qk0.txt`.

I/O behavior and TUI:
- HDF5 is loaded at runtime when possible; on failure the code writes `data.bin` instead. The console prints which HDF5 libraries were loaded (if any).
- Save progress is staged: main file [0.10..0.50], params [0.50..0.65], histories [0.65..0.80], compressed [0.80..0.90]. The status line advances accordingly, and a final "Save finished: <dir>" is printed.

Resume: automatic if a compatible checkpoint is found (version policy documented in concepts/version-compatibility.md).

## Typical runs

- Short aging run (GPU):
```bash
./RG-Evo -L 512 -l 0.5 -m 1e4 -D false
```
- CPU-only reproducibility:
```bash
./RG-Evo --gpu false -L 512 -l 0.5 -m 5e3
```
- Parameter sweep in λ around a transition:
```bash
for lam in 0.4 0.5 0.6; do ./RG-Evo -L 1024 -l "$lam" -m 2e4 -D false; done
```
