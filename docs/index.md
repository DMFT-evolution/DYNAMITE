<div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;">
	<img src="assets/logo-dmfe.svg" alt="DMFE logo" width="144" height="144"/>
	<h1 style="margin:0;">DMFE: Dynamical Mean-Field Evolution Toolkit</h1>
</div>

<p style="margin-top:0;color:#455a64;"><b>
An open-source toolkit for time-evolving non-stationary dynamical mean-field equations with memory. It targets systems that develop emergent slow time scales (e.g., quenches, aging, and other far-from-equilibrium protocols) and delivers long-time trajectories with controlled accuracy at practical cost. High performance CPU and GPU acceleration, reproducible outputs, and resumable checkpoints.
</p></b>

## Problem setting (DMFT, aging, quench)

We evolve correlation and response functions after a quench under closed dynamical equations of the form

$$
\partial_t C(t,t') = \mathcal{F}[C,R](t,t')\,,\qquad
\partial_t R(t,t') = \mathcal{G}[C,R](t,t')\,,\quad t\ge t'\,.
$$

The solver implements a numerical renormalization scheme with two-dimensional interpolation, reducing the asymptotic cost from cubic to sublinear in simulated time while controlling accuracy relevant to aging observables.

## What DMFE provides

- Non-stationary DMFT time evolution after quenches for mean-field glassy and related models.
- Two-dimensional sparse interpolation for memory kernels on the causal triangle, with controlled error.
- Adaptive high-order explicit integrators with stability-aware switching (RK54 → SSPRK104), plus sparsification-aware steps.
- CPU-first implementation with optional GPU kernels; deterministic runs and versioned outputs for reproducibility.
- Resume from checkpoints; lightweight, streaming I/O suitable for long trajectories.

When you should use it
- You study aging dynamics and two-time observables C(t, t') and R(t, t') in fully-connected/mean-field models.
- Your DMFT equations close in terms of C and R with memory integrals/convolutions over the past.
- You need long-time, high-accuracy trajectories at feasible cost (sublinear scaling in simulated time).

What DMFE assumes (scope)
- Causal, single-site effective dynamics admitting closed DMFT equations on the triangular domain t ≥ t'.
- History terms expressed as integrals/convolutions evaluable on a 2D interpolation grid.
- Model-specific closures supplied through an EOM module (examples provided). See concepts/eoms-and-observables.md.

## Quickstart

- Build Release:

```bash
./build.sh
```

- Short quench on L=512 grid:

```bash
./RG-Evo -L 512 -l 0.5 -m 1e4 -D false
```

- Outputs: HDF5 `data.h5` when available (datasets: `QKv`, `QRv`, `dQKv`, `dQRv`, `t1grid`, `rvec`, `drvec`) with attributes for parameters and time; otherwise binary with text summaries.

## Typical workflow

1) Pick or implement an EOM module defining the closures F, G for your model.
2) Choose an interpolation grid (sizes and layout) balancing accuracy and speed.
3) Select an integrator and tolerances; default adaptive RK54 usually works best initially.
4) Run a short trajectory, inspect stability and estimated errors; adjust grid or tolerances.
5) Launch production runs; use checkpoints and resume to extend time horizons.

## Where to go next

- Installation notes: install.md (toolchain/CUDA arch selection)
- Usage and flags: usage.md (physics meanings and recommended ranges)
- Equations and observables: concepts/eoms-and-observables.md
- Architecture and accuracy controls: concepts/*
- API reference (headers): reference/api/

## Outputs and formats

- Primary: HDF5 file `data.h5` with datasets `QKv`, `QRv`, `dQKv`, `dQRv` and grids `t1grid`, `rvec`, `drvec`; parameters and time stored as attributes. See usage.md for details.
- Fallback: Binary data with accompanying human-readable summaries when HDF5 is not available.
- Checkpoints: Periodic snapshots enable resume without loss of accuracy.

## Performance and accuracy

- Sparse 2D interpolation of history terms yields sublinear cost in simulated time for long runs.
- Stability-aware integrator switching maintains accuracy near stiff/transient regimes.
- Recommended starting grids are provided in concepts/interpolation-grids.md; refine based on your observable tolerances.
- CPU path is the reference; enable GPU when available for additional speedups after validating on your model.

## Cite DMFE

If this toolkit supports your research, please cite:

- Software: see Reference → Cite for release/version details (CITATION.cff).
- Method paper: J. Lang, S. Sachdev, S. Diehl, “Numerical renormalization of glassy dynamics,” arXiv:2504.06849.

License: Apache-2.0. See the License badge and the LICENSE file in the repository.

## Getting help

- See CONTRIBUTING.md and docs/dev/testing.md for guidance on reporting issues.
- Open an issue in the project tracker with a minimal input reproducing the problem and your build info (compiler/CUDA, commit hash).
