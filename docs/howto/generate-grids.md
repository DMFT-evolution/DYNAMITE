# Generate interpolation grids and metadata

This how-to walks you through generating the θ/ϕ grids, integration weights, positional maps, and interpolation metadata used by DYNAMITE.

## TL;DR

```bash
# Generate 512-point grids with barycentric Lagrange (degree 9)
./RG-Evo grid --len 512 --Tmax 100000 --dir 512 \
  --interp-method poly --interp-order 9
```

Outputs (under `Grid_data/512/`):
- `theta.dat` (N), `phi1.dat` (N×N), `phi2.dat` (N×N)
- `int.dat` integration weights (N), computed via open‑clamped B‑spline quadrature of degree s (default s=5)
- `posA1y.dat`, `posA2y.dat`, `posB2y.dat` (N×N)
- Interp metadata: `indsA1y.dat`, `weightsA1y.dat`, `indsA2y.dat`, `weightsA2y.dat`, `indsB2y.dat`, `weightsB2y.dat`

## Command reference

```bash
./RG-Evo grid [--len L] [--Tmax X] [--dir SUBDIR] \
              [--alpha X] [--delta X] \
              [--spline-order s] [--interp-method METHOD] [--interp-order n] [--fh-stencil m] [--validate]
# METHODS: poly | rational | bspline
# Short aliases: -L, -M, -d, -V, -s, -m, -o, -f
```

- `--len L` (required): grid length N. Available sets in repo: 512/1024/2048. Default 512.
- `--Tmax X`: long-time scale for θ mapping; default 100000.
- `--dir SUBDIR`: output subdirectory under `Grid_data/`. Defaults to the value of L.
- `--spline-order s`: B‑spline degree for quadrature (affects `int.dat`). Default 5.
- `--interp-method`: interpolation method for metadata.
  - `poly`: local barycentric Lagrange (degree n). Minimal n+1 weights per entry.
  - `rational`: Floater–Hormann (rational barycentric). Defaults to a single stencil (m=n+1), or set `--fh-stencil m` to blend across a wider window (m ≥ n+1).
  - `bspline`: B-spline of degree n via global collocation (dense weights).
- `--interp-order n`: interpolation degree/order (default 9).
- `--fh-stencil m`: (rational only) window size used for FH blending. Default m=n+1. Typical choices on highly irregular grids: m=n+3..n+7.
- `--validate, -V`: Do not write outputs. Recompute θ/ϕ/int with current code and compare to the saved files under `Grid_data/SUBDIR/`. Exit code 0 on success; nonzero on mismatch or missing files.

Alpha/Delta (optional):
- `--alpha` ∈ [0,1] blends a smooth non‑linear remapping of the fractional index into the default mapping; `--delta ≥ 0` sets the softness around the center. Defaults are 0 (paper‑exact grid). If set, the values are recorded in `Grid_data/<subdir>/grid_params.txt`.

## Choosing a method

- Use `poly` (or `rational`) when the interpolated data change between calls. They store exactly n+1 weights per output entry and evaluate fast.
- Use `rational` with a wider `--fh-stencil` to improve stability on highly irregular nodes while keeping exactly m local weights per entry.
- Use `bspline` when you specifically need spline smoothness/derivatives and can amortize the global solve or reuse weights. It writes dense per-entry weights.

## File formats

- `inds*.dat`: N×N TSV of integers. For local methods, each entry is the start index of a contiguous stencil in θ. For `bspline`, entries are -1.
- `weights*.dat`: vector of weights, one block per output entry in row-major order.
  - Local methods: each block has n+1 weights.
  - B-spline: each block has N weights (global).

## Tips

- Start with `--interp-method poly --interp-order 9` (these are the defaults).
- Keep `--Tmax` consistent across L when comparing convergence.
- The generated files are read at runtime; ensure `Grid_data/<L>/` is on the target machine.
