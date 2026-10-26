# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/grid.svg" alt="Grid icon"/> Tutorial: Generate grids

The program reads the $\theta$ grid and the precomputed interpolation and quadrature weights from `Grid_data/<L>/` in the current working directory. The repository contains `Grid_data/512/`, built for times up to $t_{\max}=10^5$. You need a new grid if you want

- a different number of points $L$, for example to check convergence, or
- runs beyond $t\approx10^5$.

## A larger grid

```bash
./RG-Evo grid -L 1024
```

writes `Grid_data/1024/` in about ten seconds (0.9 GB). Afterwards `./RG-Evo -L 1024 ...` uses it. If you forget this step, the program generates the grid itself on the first run with `-L 1024`.

## A grid for long runs

The grid has to resolve the microscopic time scales up to the latest simulated time. For runs up to $t=10^7$:

```bash
mv Grid_data/512 Grid_data/512-tmax1e5      # keep the shipped grid
./RG-Evo grid -L 512 -M 1e7
```

`-M` sets $t_{\max}$; it should not be smaller than the final time `-t` of your runs. The simulation always reads `Grid_data/<L>/`, so only one grid per $L$ is active at a time; keep others under different names.

## What is written

`Grid_data/<L>/` contains the nodes (`theta.dat`), the interpolation points and weights (`phi*.dat`, `pos*.dat`, `inds*.dat`, `weights*.dat`), the quadrature weights (`int.dat`) and `grid_params.txt` with the parameters used. Each run copies these parameters into its `params.txt`, so it is always clear which grid a result was obtained with. The files are described in [Grid and interpolation](../concepts/interpolation-grids.md).

## Other choices

The default interpolation (Lagrange polynomials of order 9 in the grid index) is a good choice in nearly all cases. All options, including other interpolation methods, a user-supplied $\theta$ grid and the check `-V` of existing files, are listed in the [Grid generator](../grid-generator.md) reference.
