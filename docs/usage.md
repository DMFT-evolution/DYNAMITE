# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/algorithm.svg" alt="Usage icon"/> Usage

This page is your **practical starting point** for running DYNAMITE: CLI flags, inputs/outputs, and common workflows.

For the algorithmic background and implementation concepts, continue with:

- **Concepts → Algorithm**
- **Concepts → Interpolation grids**
- **Concepts → Time integration**
- **Concepts → EOMs and observables**

Entry point: `./RG-Evo`. Show all options with `-h`.

## Model and parameters

- `-p`, `-q` (integers): model orders (e.g., spherical mixed p-spin). Use `p>=2`; common: `p=3, q=4`.
- `-l, --lambda` (float): coupling strength (dimensionless). Typical 0–1; study transitions by sweeping.
- `-T, --T0` (float|inf): initial temperature; use `inf` for zero-noise quenches.
- `-G, --Gamma` (float): final temperature.

Note: At present, the mixed spherical p-spin equations are hardcoded as in the accompanying paper; other models are not yet implemented. Modular equations of motion will be subject of a future release.

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
- `-R, --log-response-interp` boolean: interpolate the response $QR$ in log space (default false). If enabled, the code interpolates $f_R=\log(QR)$ with $g_R=dQR/QR$ and exponentiates back (fallback to linear when any stencil has $QR\le 0$). The correlation $QK$ is interpolated in the ordinary linear domain.

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

If `Grid_data/<L>/` is missing (or you want to regenerate with different settings), use the built-in grid generator.

For the full `./RG-Evo grid` flag reference, outputs, and validation workflow, see
[How-to → Generate new grids](howto/generate-grids.md).

If you just want the default grids for a first run, see
[Tutorial → Generate grids](tutorials/generate-grids.md).

## Outputs and observables

Each run writes a self-contained output directory that contains:

- the full state (preferably `data.h5`, otherwise `data.bin`),
- a provenance record (`params.txt`),
- and lightweight text summaries for quick plotting.

For a practical guide to locating files and doing sanity checks, see
[Tutorial → Reading outputs](tutorials/reading-outputs.md).

For the on-disk formats (including the compressed snapshot files), see
[Concepts → Architecture](concepts/architecture.md).

Resume is automatic if a compatible checkpoint is found; see
[Concepts → Version compatibility](concepts/version-compatibility.md).

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
