# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/cpu.svg" alt="CPU icon"/> Tutorial: Running without a GPU

DYNAMITE does not need a GPU. The CPU backend solves the same equations with the same grids, interpolation and step-size control, and gives the same results to within the tolerance. A GPU makes long runs considerably faster, but for exploring parameters, for times up to $10^3$ or $10^4$, and for learning how the code behaves, an ordinary workstation is enough.

## Build

```bash
./build.sh --cuda=off
```

builds without CUDA. You need a C++17 compiler and CMake; install OpenMP as well, otherwise the program runs on a single core (see [Install](../install.md)). A CUDA build can also be used: on a machine without a usable GPU it switches to the CPU by itself, and `-g false` selects the CPU explicitly.

## Run

```bash
export OMP_NUM_THREADS=8          # number of cores to use
./RG-Evo -p 3 -q 4 -l 0.5 -T inf -G 0 -t 1000 -m 1e7 -g false -D false -o Results/
```

The work of each time step is shared among `OMP_NUM_THREADS` threads; without this variable, OpenMP usually takes all cores. The memory integrals are distributed over the threads, while some smaller parts of a step run serially.

## What to expect

Measured on a desktop computer with an Intel Core i7-13700 (8 threads) and, for comparison, an NVIDIA RTX 4090, for the mixed 3+4 model ($\lambda=1/2$) quenched from $T_0=\infty$ to $T=0$ with $L=512$ and the default tolerance:

| Final time | Steps | CPU, 8 threads | GPU |
|---|---|---|---|
| $t=1000$ | 21 000 | 17 min | 47 s |

A step costs about 0.05 s on the CPU, independently of how long the run has been going. At late times the step size approaches its stability limit, so the number of steps, and hence the run time, then grows linearly with $t$. For this model the step size cannot exceed about 0.25 with the Dormand–Prince method, so continuing the run above to $t=10^4$ takes at least another half hour on the CPU, and runs to $10^6$ or $10^7$, as in the papers, need around a million steps or more and are a job for a GPU.

The memory needed grows with the number of stored time slices: the run above used 1.5 GB. Sparsification (see [Sparsification](../concepts/sparsification.md)) limits this in long runs.

## Things to keep in mind

- **Reproducibility.** With more than one thread, the order of the parallel sums varies from run to run, so two identical CPU runs can differ at the level of rounding errors. With `OMP_NUM_THREADS=1` a CPU run is reproducible bit for bit.
- **Continuing runs.** A run can be stopped and continued later: repeating the command with a larger `-t` continues from the last checkpoint (see [Continuing a run](../usage.md#continuing-a-run)). Checkpoints are written every `--sparsify-interval` steps (100000 by default); for interrupted CPU runs a smaller value, e.g. `--sparsify-interval 10000`, loses less work.
- **Smaller grids.** For quick surveys, a grid with fewer points, e.g. $L=256$ (`./RG-Evo grid -L 256`), reduces the cost per step by about a factor four. Check the results you keep with $L=512$.
