# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/architecture.svg" alt="Architecture icon"/> Code structure and file formats

## Source tree

Headers are in `include/<module>/`, sources in `src/<module>/`. Most modules have a CPU implementation (`.cpp`) and a CUDA implementation (`.cu`) with the same interface; the CUDA files are only compiled in CUDA builds.

| Module | Content |
|---|---|
| `EOMs` | right-hand sides of the equations of motion (`time_steps`), Runge–Kutta methods (`runge_kutta`) |
| `math` | the covariance $f$ and its derivatives (`math_ops`), the memory kernels (`math_sigma`) |
| `interpolation` | interpolation in $\theta$ and $t$ at the quadrature nodes |
| `convolution` | quadrature sums for the memory integrals |
| `search` | location of interpolation intervals in the adaptive $t$ grid |
| `sparsify` | removal of time slices from the history |
| `simulation` | main loop, step-size control, rollback |
| `grid` | the `RG-Evo grid` subcommand |
| `io` | checkpoints, `params.txt`, text output, loading of grids and checkpoints |
| `core` | command-line options, initialization, CPU/GPU dispatch, memory management |
| `ui`, `version` | terminal output; version information and compatibility checks |

`main.cpp` parses the options, either runs the grid generator or initializes a simulation, and calls the main loop in `src/simulation/simulation_runner.cpp`. The build produces a static library `DFME-core` and the executable `RG-Evo`.

## Output directory

Each run directory (see [Usage](../usage.md#output)) contains:

| File | Written | Content |
|---|---|---|
| `params.txt` | at every checkpoint | run record: command line, all parameters, program version and Git commit, compiler and CUDA version, host name, CPU or GPU mode, grid parameters, run time and peak memory |
| `data.h5` or `data.bin` | at every checkpoint | the full state, from which a run can be continued |
| `correlation.txt` | after every step | columns $t$, $E(t)$, $C(t,0)$ |
| `QK_compressed`, `QR_compressed`, `t1_compressed.txt` | at every checkpoint | $C$ and $R$ at the last time on an $L\times L$ grid, see below |
| `energy.txt`, `rvec.txt`, `qk0.txt` | at every checkpoint | $E(t)$, $\mu(t)$ and $C(t,0)$ at the stored times $t_n$ |
| `step_metrics.txt` | at every checkpoint, with `-D true` | per step: $t$, wall-clock time, memory used |

Checkpoints are written at every maintenance pass (every `--sparsify-interval` steps) and at the end of the run. `correlation.txt` is the most complete record of a run, because it contains every step, also those removed later by sparsification; the files written at checkpoints only contain the stored times. It also keeps the steps that were later discarded by a rollback: after a rejected step the time in the first column jumps back. To remove the discarded steps, keep only the lines whose time is smaller than the time of every later line (see [Reading outputs](../tutorials/reading-outputs.md)). Text files start with a comment line beginning with `#` (except `correlation.txt`).

### Checkpoint

`data.h5` holds the datasets `QKv`, `QRv`, `dQKv`, `dQRv` (length $L\times n_t$, time-major: the $L$ values of slice $n$ are contiguous), `t1grid`, `rvec`, `drvec` (length $n_t$), and attributes with the model and numerical parameters, the current time, step count and energy. The meaning of the datasets is given in [Equations](eoms-and-observables.md#what-is-stored). `data.bin` contains the same data in a binary format that is read by the program itself. `scripts/ImportDMFE.wl` imports `data.h5` into Mathematica.

### Compressed snapshot

`QK_compressed` and `QR_compressed` each hold an $L\times L$ matrix: two integers of type `size_t` (rows, columns), followed by the values as `double` in row-major order, all in the byte order of the machine that wrote them. With $t_i=\theta_i\,t_{\rm last}$ listed in `t1_compressed.txt`,

$$
\texttt{QK\_compressed}[i,j]=C(t_i,\theta_j t_i)\,,\qquad \texttt{QR\_compressed}[i,j]=R(t_i,\theta_j t_i)\,,
$$

i.e. row $i$ is the correlation at the earlier time $t_i$, as a function of $\theta=t'/t_i$. To read off $C(t_w+\tau,t_w)$, interpolate at $t_i=t_w+\tau$ and $\theta=t_w/(t_w+\tau)$. `scripts/plot_correlation.py` does this; see [Reading outputs](../tutorials/reading-outputs.md).

## CPU and GPU

The program is a single executable. In a CUDA build, `-g true` (the default) runs the integration on the GPU if one is found, and otherwise on the CPU. The state lives in host memory (`SimulationData::host`) or device memory (`SimulationData::device`); for checkpoints the device data is copied to the host and written, by default in a background thread (`-A true`), so that the integration continues meanwhile. At most one checkpoint is written at a time.

The two backends implement the same equations, interpolation, quadrature and step-size control. They differ in the order of the floating-point operations and therefore agree only to within the tolerance, not bit for bit. Remaining differences in the algorithm: the SERK2 methods exist only on the GPU, and the sparsification considers every second slice on the GPU but all slices (with the same rule against removing neighbours) on the CPU.
