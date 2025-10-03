# DMFE: Dynamical Mean-Field Evolution Toolkit

DMFE solves non-stationary dynamical mean-field equations for glassy and related systems with GPU acceleration and a validated CPU path. It targets expert users in statistical physics, condensed matter, and machine learning working with aging dynamics and DMFT-like closures.

## Problem setting (DMFT, aging, quench)

We evolve correlation and response functions after a quench under closed dynamical equations of the form

$$
\partial_t C(t,t') = \mathcal{F}[C,R](t,t')\,,\qquad
\partial_t R(t,t') = \mathcal{G}[C,R](t,t')\,,\quad t\ge t'\,.
$$

The solver implements a numerical renormalization scheme with two-dimensional interpolation, reducing the asymptotic cost from cubic to sublinear in simulated time while controlling accuracy relevant to aging observables.

## Method at a glance

- Two-dimensional sparse interpolation for history terms
- Adaptive RK54 as default; auto-switch to SSPRK104 at stability limit; optional SERK2 trials after sparsification
- GPU kernels with portable CPU fallback
- Asynchronous I/O; resume runs and version-compat checks

Relevant modules: core, EOMs, interpolation, convolution, sparsify, simulation, io.

## Quickstart (expert)

- Build Release:

```bash
./build.sh
```

- Short quench on L=512 grid:

```bash
./RG-Evo -L 512 -l 0.5 -m 1e4 -D false
```

- Outputs: HDF5 `data.h5` when available (datasets: `QKv`, `QRv`, `dQKv`, `dQRv`, `t1grid`, `rvec`, `drvec`) with attributes for parameters and time; otherwise binary with text summaries.

## Where to go next

- Installation notes: install.md (toolchain/CUDA arch selection)
- Usage and flags: usage.md (physics meanings and recommended ranges)
- Equations and observables: concepts/eoms-and-observables.md
- Architecture and accuracy controls: concepts/*
- API reference (headers): reference/api/
