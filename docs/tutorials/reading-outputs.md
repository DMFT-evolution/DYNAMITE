# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/file.svg" alt="File icon"/> Tutorial: Reading outputs (and trusting them)

DYNAMITE writes a **self-contained output directory** for each run. This directory is designed to be both:

- easy to inspect quickly (text summaries), and
- reproducible / restartable (versioned params + full state in HDF5 or binary).

## What files to expect

Depending on the build and available libraries you’ll get **either**:

- `data.h5` (preferred when HDF5 is available), **or**
- `data.bin` (binary fallback, always available),

and additionally:

- `params.txt` (parameters + version/build/provenance)
- text summaries such as `energy.txt`, `correlation.txt`, `rvec.txt`, `qk0.txt` (exact set may depend on run options)

If you build without HDF5 support (or runtime HDF5 loading fails), the code automatically falls back to `data.bin` and still produces the same core state.

## Quick inspection (HDF5)

If `data.h5` exists, you can inspect structure with common tools:

```bash
h5ls -r data.h5
h5dump -n data.h5
```

Core datasets written by DYNAMITE include:

- `QKv`, `QRv`: correlation/response samples on the internal (non-equidistant) time-ratio grid
- `dQKv`, `dQRv`: time derivatives (used by integrators / diagnostics)
- `t1grid`: the (adaptive) time nodes $t$ used during integration
- `rvec`, `drvec`: diagonal/reduced observables tracked during the run

Tip: `params.txt` is the fastest way to see **exactly** what the run did (grid size, tolerances, CPU/GPU mode, etc.).

## Text summaries (fast plotting)

For most physics-facing diagnostics you don’t need to parse the full 2D state immediately. Start with the text files:

- `energy.txt`: time series of the energy (used for stability checks and aging analysis)
- `correlation.txt`: commonly used slices/diagnostics of $C$ (format documented in-file; see file header)
- `rvec.txt`: reduced/diagonal observables (format documented in-file)

These are intended for quick plotting and sanity checks.

## Binary fallback (`data.bin`) is fully supported

`data.bin` is not a “degraded mode”: it’s the supported non-HDF5 carrier for the full simulation state.

- Save path: when HDF5 is unavailable (or fails), the program writes `data.bin` instead.
- Load/resume path: binaries can be loaded to resume trajectories if compatible with the current build/version policy.

Implementation references (for developers): see declarations in `include/io/io_utils.hpp` and implementations under `src/io/` (binary save/load, plus optional HDF5 writers).

## Reproducibility & provenance (how to trust a run)

Each output directory includes the information needed to reproduce a run:

1) **Exact code identity**

Open `params.txt` and record:
- `code_version`, `git_hash`, `git_branch`, `git_tag`, `git_dirty`  
- compiler/CUDA versions and build timestamp

2) **Exact runtime configuration**

In `params.txt` you’ll also find:
- the full stored command line  
- physical parameters ($p$, $p2$, $\lambda$, $T_0$, $\Gamma$)  
- numerical parameters (grid `len`, tolerances, sparsification settings, integrator toggles)

3) **Grid provenance**

The run uses precomputed grid packages under `Grid_data/<L>/`. The grid generator writes metadata to:

- `Grid_data/<L>/grid_params.txt`

and DYNAMITE mirrors key entries into `params.txt` (prefixed with `grid_...`). This makes it easy to confirm that a run used the intended grid package even after you move/copy the output directory.

Practical recommendation: keep `params.txt` alongside `data.h5`/`data.bin` when archiving or sharing results.

## Sanity checks before trusting long runs

These checks are fast and catch most common issues:

- **CPU vs GPU short-time agreement**: run a short trajectory with `--gpu false` and compare key summaries (e.g. `energy.txt`).
- **Grid convergence**: compare L=512 vs 1024 (and 2048 if needed) at fixed parameters.
- **Tolerance sensitivity**: tighten the integrator tolerance `-e` and confirm observables don’t shift materially.
- **Sparsification sensitivity (spot check)**: run briefly with `--sparsify-sweeps 0` and confirm agreement in your observables of interest.
- **Resume discipline**: when resuming, confirm `params.txt` compatibility and keep the original directory intact.

## Next: plotting scripts

We provide small Python helpers under `scripts/` to quickly plot standard summaries (energy vs time, etc.). See:

- `scripts/plot_energy.py`
- `scripts/plot_text_series.py`

These are intentionally lightweight and can be adapted to create publication figures (e.g. $C(t_w+\tau, t_w)$ at multiple waiting times, and response-vs-correlation parametric plots).

Dependencies: these scripts use `numpy` and `matplotlib`.
