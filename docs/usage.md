# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/algorithm.svg" alt="Usage icon"/> Usage

```bash
./RG-Evo [options]          # run a simulation
./RG-Evo grid [options]     # generate interpolation grids, see Grid generator
```

`./RG-Evo -h` prints all options with their current defaults. Boolean options take `true` or `false`.

## Model

The program solves the dynamical mean-field equations of the mixed spherical $p$-spin model with covariance

$$
f(q)=\lambda\,q^{p}+(1-\lambda)\,q^{s},
$$

for a system prepared in equilibrium at temperature $T_0$ and coupled at $t=0$ to a bath at temperature $T$. The equations are given in [Equations](concepts/eoms-and-observables.md).

| Option | Default | Meaning |
|---|---|---|
| `-p INT` | 3 | First interaction order $p$. |
| `-q INT` | 12 | Second interaction order $s$ (called `p2` in `params.txt`). |
| `-l, --lambda F` | 0.3 | Weight $\lambda$ of the $p$-body term; $\lambda=1$ is the pure $p$-spin model. |
| `-T, --T0, --initial-temperature F` | `inf` | Temperature $T_0$ of the initial equilibrium state. It enters only through $\beta_0=1/T_0$; `inf` gives a random initial condition ($\beta_0=0$). |
| `-G, --Gamma, --bath-temperature F` | 0 | Bath temperature $T$ after the quench. Despite the name `Gamma`, this is a temperature, not a damping rate. `-G 0` is a quench to zero temperature. |

## Accuracy and length of the run

| Option | Default | Meaning |
|---|---|---|
| `-L INT` | 512 | Number $L$ of grid points in $\theta=t'/t$. Selects `Grid_data/<L>/`. |
| `-e, --error F` | `1e-10` | Tolerance $\epsilon$ for the Runge–Kutta error estimate. |
| `--error-norm STR` | `sum` | How the error estimate is formed from its $2L$ components $\delta C(t,\theta_i t)$, $\delta R(t,\theta_i t)$: `sum` (sum of absolute values), `mean` (`sum` divided by $2L$), `rms` (root mean square) or `max` (largest component). |
| `-t F` | `1e7` | Final time. |
| `-m INT` | 10000 | Maximal number of time steps. A continued run counts the steps of the earlier run as well. |
| `-d F` | `1e-5` | Initial and smallest time step. The run stops if the step size falls below it. |

The run ends when it reaches the time `-t`, the step count `-m`, or the smallest step `-d`, whichever comes first.

With `--error-norm sum`, the default, the error estimate grows with $L$: at a fixed step size it is roughly proportional to $L$, so the same `-e` is stricter on a larger grid. `mean` and `rms` do not depend on $L$. Since `-e` is compared with the chosen norm, a different norm changes the meaning of a given `-e`; with `mean`, for instance, `-e 1e-10` corresponds to `-e 1e-7` with `sum` at $L=512$. All results in the papers use `sum`.

How the time step is chosen is described in [Time integration](concepts/time-integration.md).

## Integrators and history reduction

| Option | Default | Meaning |
|---|---|---|
| `-S, --serk2 BOOL` | `true` | GPU only: allow the second-order SERK2 schemes after SSPRK(10,4) once the step size permits. Has no effect on the CPU backend. |
| `-w, --sparsify-sweeps INT` | `-1` | Number of sparsification sweeps per maintenance pass. `0` switches sparsification off; `-1` means one sweep, or two on a GPU whose memory is more than half full. |
| `--sparsify-interval INT` | 100000 | Number of time steps between two maintenance passes. A pass sparsifies the stored history, recomputes the interpolation and writes a checkpoint. |
| `-R, --log-response-interp BOOL` | `false` | Interpolate $\log R$ instead of $R$ in $t$. Falls back to $R$ wherever a stencil contains non-positive values. |
| `-F, --tail-fit BOOL` | `false` | Experimental: after each step, blend $C$ near $\theta=1$ with a fit of $(1-C)/(1-\theta)^2$. |

With `-w -1` the result can depend on how much GPU memory other processes use. For runs that must be reproducible, set the number of sweeps explicitly, e.g. `-w 1`. See [Sparsification](concepts/sparsification.md).

## Hardware

| Option | Default | Meaning |
|---|---|---|
| `-g, --gpu BOOL` | `true` | Use the GPU if one is available. Without a usable GPU the program falls back to the CPU and says so. |

On the CPU the work is shared among OpenMP threads; set their number with the environment variable `OMP_NUM_THREADS`. With more than one thread, two identical CPU runs can differ at the level of rounding errors, because the order of the parallel sums varies. Runs with `OMP_NUM_THREADS=1` are reproducible bit for bit; so are GPU runs. See [Running without a GPU](tutorials/cpu-only.md).

## Output

| Option | Default | Meaning |
|---|---|---|
| `-o, --out-dir DIR` | see below | Directory in which the run directories are created. |
| `-s BOOL` | `true` | Write output. With `false` nothing is written to disk. |
| `-A, --async-export BOOL` | `true` | Write checkpoints in the background while the integration continues. |
| `-D BOOL` | `true` | Verbose output: one line per step, timing and memory per step in `step_metrics.txt`, and additional checks. On the GPU this makes the run noticeably slower; use `-D false` for production runs. |

Every run writes into its own directory, named after the date, time and parameters, for example

```
Results/26_10_2026_1534_p=3_q=4_lambda=0.50_T0=inf_G=0.00_L=512/
```

The directory contains the checkpoint (`data.h5` or `data.bin`), the run record `params.txt`, and text files for plotting; see [Reading outputs](tutorials/reading-outputs.md).

**Where the run directories go.** With `-o DIR` they are created in `DIR`. Without `-o` they are created in `Results/` below the current working directory. (A build configured with the CMake option `DMFE_HOME_OUTPUT_DIR` uses that directory instead whenever the executable lies inside your home directory; see [Install](install.md).)

## Continuing a run

Before creating a new run directory, the program searches `Results/` and the output directory for an earlier run with the same parameters: `-p`, `-q`, `-l`, `-T`, `-G`, `-L`, `-d`, `-e`, `-S`, and, if recorded, `-R`, `-F`, `-w`, `--sparsify-interval`, `--error-norm` and the grid. If it finds one written by a compatible program version, it loads the checkpoint and continues that run in the same directory.

So, to extend a finished run to later times, repeat the command with a larger `-t` (and `-m`). To start afresh with the same parameters, use a different `-o` or move the old directory away. Which program versions count as compatible is explained in [Version compatibility](concepts/version-compatibility.md).

## Grids

The interpolation grids are read from `Grid_data/<L>/` in the current working directory. Only `Grid_data/512/` is part of the repository. If the directory for the requested `-L` does not exist, the program first looks for another directory in `Grid_data/` that contains a grid of the same length, and otherwise runs `./RG-Evo grid -L <L>` to generate one. For $L=512$ this takes a few seconds and about 220 MB of disk space; the size grows as $L^2$. See [Generate grids](tutorials/generate-grids.md) and the [Grid generator](grid-generator.md) reference.

## Other options

| Option | Meaning |
|---|---|
| `-I, --allow-incompatible-versions BOOL` | Also continue runs written by an incompatible program version (see [Version compatibility](concepts/version-compatibility.md)). |
| `-c, --check FILE` | Report whether the run record `FILE` (a `params.txt`) is compatible with this program version, then exit. |
| `-v` | Print version and build information, then exit. |
| `-h, --help` | Print all options with their defaults, then exit. |

## Examples

```bash
# Pure 3-spin model, quench from T0 = inf to T = 0, up to t = 1000
./RG-Evo -p 3 -q 4 -l 1 -T inf -G 0 -t 1000 -m 1e6 -D false -o Results/

# Mixed 3+4 model, quench from T0 = 1.0 to T = 0, on the CPU with 8 threads
OMP_NUM_THREADS=8 ./RG-Evo -p 3 -q 4 -l 0.5 -T 1.0 -G 0 -t 100 -m 1e6 -g false -o Results/

# Same parameters with a finer grid and a tighter tolerance, as a convergence check
./RG-Evo -p 3 -q 4 -l 0.5 -T 1.0 -G 0 -t 100 -m 1e6 -L 1024 -e 1e-11 -o Results/
```
