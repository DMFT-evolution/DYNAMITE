# DYNAMITE

[![Docs](https://img.shields.io/badge/docs-website-blue)](https://dmft-evolution.github.io/DYNAMITE/) [![License](https://img.shields.io/badge/license-Apache--2.0-orange)](LICENSE) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23679051.svg)](https://doi.org/10.5281/zenodo.23679051)

DYNAMITE (DYNAmical Mean-fIeld Time Evolution solver) integrates two-time dynamical mean-field equations for the correlation and response functions $C(t,t')$ and $R(t,t')$ to long times. The released program solves the equations of the mixed spherical $p$-spin model after a quench and reaches times of order $10^7$.

The two-time functions are stored on a fixed non-uniform grid in $\theta=t'/t$ and an adaptive grid in $t$. The memory integrals are evaluated by interpolation with precomputed weights, and time slices that the interpolation does not need are removed from the history. Each time step then has a fixed cost, so that the run time grows linearly with the simulated time at late times, instead of cubically as with a uniform time step. The program runs on CPUs and, considerably faster, on NVIDIA GPUs.

**Documentation:** <https://dmft-evolution.github.io/DYNAMITE/>

## Build

Requirements: CMake ≥ 3.24 and a C++17 compiler; OpenMP, CUDA and HDF5 are optional.

```bash
./build.sh               # with CUDA if available
./build.sh --cuda=off    # CPU only
```

The executable `RG-Evo` is placed in the repository root. See [Install](https://dmft-evolution.github.io/DYNAMITE/install/) for build options.

## Run

```bash
./RG-Evo -h                                   # all options and defaults
./RG-Evo -p 2 -q 3 -l 1 -T inf -G 0 -e 1e-8 -t 1 -m 1e6 -g false -o Results/
```

This runs the pure 2-spin model (`-l 1`), quenched from a random initial condition (`-T inf`) to zero temperature (`-G 0`), up to $t=1$ on the CPU. The results are written to a new directory below `Results/`; `correlation.txt` lists $t$, the energy $E(t)$ and $C(t,0)$. Interpolation grids are read from `Grid_data/<L>/` (the repository contains $L=512$) and generated automatically for other $L$.

See the [first run](https://dmft-evolution.github.io/DYNAMITE/tutorials/first-run/) and [running without a GPU](https://dmft-evolution.github.io/DYNAMITE/tutorials/cpu-only/) tutorials.

## Citation

If you use DYNAMITE, please cite

J. Lang, V. Citro, L. Leuzzi, F. Ricci-Tersenghi, “DYNAMITE: A high-performance framework for solving Dynamical Mean-Field Equations,” [arXiv:2604.06309](https://arxiv.org/abs/2604.06309).

The method was introduced in J. Lang, S. Sachdev, S. Diehl, “Numerical renormalization of glassy dynamics,” Phys. Rev. Lett. **135**, 247101 (2025), [doi:10.1103/z64g-nqs6](https://doi.org/10.1103/z64g-nqs6).

## Contributing and license

See [`CONTRIBUTING.md`](CONTRIBUTING.md). DYNAMITE is distributed under the Apache License 2.0; see [`LICENSE`](LICENSE).
