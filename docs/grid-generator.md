# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/grid.svg" alt="Grid icon"/> Grid generator

```bash
./RG-Evo grid [options]
```

computes the $\theta$ grid, the interpolation weights and the quadrature weights and writes them to `Grid_data/<dir>/` below the current working directory. The meaning of the grid and of the files is explained in [Grid and interpolation](concepts/interpolation-grids.md); a short walk-through is in [Generate grids](tutorials/generate-grids.md).

## Options

| Option | Default | Meaning |
|---|---|---|
| `-L, --len N` | 512 | Number of grid points $L$. |
| `-M, --Tmax X` | `1e5` | Largest time $t_{\max}$ the grid is designed for. Should be at least the final time of the runs. |
| `-d, --dir SUBDIR` | `<len>` | Output directory below `Grid_data/`. A simulation with `-L N` reads `Grid_data/N/`. |
| `-m, --interp-method M` | `index-poly` | Interpolation in $\theta$: `index-poly`, `index-rational`, `index-hermite`, `index-bspline`, `poly`, `rational`, `bspline`. |
| `-o, --interp-order n` | 9 | Interpolation order; the stencil has $n+1$ points. |
| `-f, --fh-stencil m` | $n+1$ | Number of points for the rational (Floater–Hormann) methods, $m\ge n+1$. A larger $m$ is less sensitive to noise. |
| `-s, --spline-order n` | 5 | Order of the spline used for the quadrature weights. |
| `-I, --int-method M` | `index-spline` | Quadrature weights from a spline in the grid index (`index-spline`) or directly in $\theta$ (`theta-spline`). |
| `-a, --alpha X` | 0 | Reshaping of the grid, $0\le\alpha\le1$. |
| `-D, --delta X` | 0 | Reshaping of the grid, $\delta\ge0$. |
| `-T, --theta-file FILE` | | Read the nodes $\theta_i$ from `FILE` instead (one per line, strictly increasing, from 0 to 1). This implies `--int-method theta-spline` and an interpolation method in $\theta$; `index-*` is replaced by `poly`. |
| `-V, --validate` | | Write nothing; recompute $\theta$, $\phi^{(1,2)}$ and the quadrature weights and compare them with the files in `Grid_data/<dir>/`. Exit status 0 if they agree. |

Note that the option letters of the grid generator differ from those of a simulation: `-d` is the directory here, `-D` the parameter $\delta$.

## Examples

```bash
# The shipped grid
./RG-Evo grid -L 512

# A grid for runs up to t = 1e7
./RG-Evo grid -L 512 -M 1e7

# L = 1024, written to Grid_data/1024/
./RG-Evo grid -L 1024 -M 1e7

# Floater-Hormann interpolation of order 9 on 15 points, in a separate directory
./RG-Evo grid -L 512 -m index-rational -o 9 -f 15 -d 512-fh

# Check that the files in Grid_data/512/ agree with the current generator
./RG-Evo grid -L 512 -V
```

A simulation always reads `Grid_data/<L>/`. To use a grid stored under another name, rename the directory, or remove `Grid_data/<L>/`: the program then looks for another directory in `Grid_data/` with a grid of the same length and links to it.

## Time and disk space

For $L=512$ the generator takes a few seconds and writes about 220 MB. The files grow as $L^2$ (about 0.9 GB for $L=1024$), and the time somewhat faster.
