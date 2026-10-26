# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/); the project
uses four-component version tags `vMAJOR.MINOR.PATCH.BUILD`.

Commit hashes are given for every tag so that any published result can be tied to
an exact revision.  Tags up to `v1.1.0.3` are annotated tags; the hash quoted is
the *commit* the tag points at, not the tag object.

## [1.4.0.0] - first release
Release accompanying the revised version of the manuscript
"DYNAMITE: A high-performance framework for solving Dynamical Mean-Field Equations"
(Ms. COMPHY-D-26-00359).  All changes are marked in the source with a `[CPC-REV]`
comment.

### Fixed
- **CPU Dormand-Prince integrator (`RK54`) evaluated the stage derivatives at the
  stage-zero state.**  Only the pointwise part of the right-hand side was affected
  (`-mu C`, `-mu R`, the `beta_0` forcing and the `2 T R` bath term), but that is
  enough to reduce the CPU integrator from fifth to *first* order globally, while
  the embedded error estimator remained blind to it because
  `sum_n (b_n - b2_n) = 0` exactly.  Measured on the pure `p=2` model
  (`L=512`, `-e 1e-6`): `C(t,0)` changed by `1.2e-6` at `t = 0.126` and by
  `8.0e-4` (0.17 % relative, ~800 x the requested tolerance) at `t = 1.46`.
  The GPU integrators and the CPU `SSPRK(10,4)` were unaffected.
  (`src/EOMs/runge_kutta.cpp`)
- **Two non-equivalent implementations of `sparsifyNscale()`.**  CUDA builds
  compiled the one in `sparsify_utils.cu`, CPU-only builds the one in
  `sparsify_utils.cpp`; the latter had no alternation guard, was shifted by one
  index and, most importantly, did not remap the stored derivatives
  (`indsD`/`tfac`) after a removal.  There is now a single host implementation,
  compiled in both configurations.  (`src/sparsify/`)
- **GPU energy-history kernel (latent).**  `computeEnergyHistoryKernel` still
  evaluated the `beta_0` contribution of the energy as `f'(C(t,0))/T0` instead of
  `f(C(t,0))/T0`, and multiplied it by `t` -- inconsistent with Eq. (14) of the
  paper.  The CPU path and the scalar `energyGPU()` had been corrected in
  `de696f84`; this kernel was missed.  Measured impact: none.  On the code paths
  we could exercise on an H100, `energy.txt` is produced by the host routine and
  the kernel is not reached (released and revised binaries give a bit-for-bit
  identical `energy.txt` at `T0 = 1` over 1264 nodes).  (`src/io/io_output.cu`)
- **ODR violation between the two backend dispatchers.**
  `src/core/backend_dispatch.cpp` defines `update()`, `energy()`,
  `initializeGPUBackend()` and `synchronizeCompressedData()` with no preprocessor
  guard -- the same four symbols as `backend_dispatch.cu` -- and CMakeLists listed
  it in `CPU_SOURCES` unconditionally, so in a CUDA build both objects entered the
  same static library and the choice was the linker's.  Had the CPU one won,
  `initializeGPUBackend()` would have reported "program was built without CUDA
  support" on a machine with a working GPU.  Now compiled only when CUDA is off.
  (`CMakeLists.txt`)
- **Runge-Kutta error estimate contained a pure round-off term.**  `|gt - gte|` is
  analytically zero (`sum b = sum b2 = 1`), but was accumulated separately and grew
  like `ulp(t)`, reaching `~2e-8` at `t = 1e7` -- two orders of magnitude above the
  default tolerance.  Removed.  (`src/EOMs/runge_kutta.{cpp,cu}`)
- **CPU interpolation in t in the last interval.**  `indexVecN()` (CPU), which
  interpolates the history at the times `theta_i t` for the first memory integral,
  omitted the derivative term in the last interval, i.e. between the last stored slice
  and the current Runge-Kutta stage, where the GPU kernel uses the quadratic Hermite
  interpolant with the left derivative.  The CPU solution was therefore several orders
  of magnitude less accurate than the GPU one, and its error estimate correspondingly
  larger (pure `p=2` model, `-e 1e-8`, up to `t=1`: max |C - C_exact| = 7e-9 with
  6640 steps on the CPU, 2e-13 with 1263 steps on the GPU; after the fix the CPU gives
  2e-13 with 1271 steps).  The log-response branch (`-R true`) of the same routine used
  `log(dQRv)` instead of `dQRv/QRv` and wrong signs in the last interval, and the
  routine read `dtratio` out of bounds.  It now follows the GPU kernel branch by branch;
  the intermediate quantities of a Runge-Kutta stage agree between the backends to
  rounding error.  (`src/interpolation/index_vec.cpp`)
- **CPU energy in `correlation.txt` with a finite `T0`.**  `energyCPU()` took C(t,0)
  from the last entry of the last slice, which is C(t,t) = 1, instead of the first.
  With a finite initial temperature the energy written after every step on the CPU
  therefore contained -f(1)/T0 instead of -f(C(t,0))/T0 (e.g. -1 instead of -0.5 for a
  run started in equilibrium at T0 = T = 2).  The GPU routine and the energy written to
  `energy.txt` were correct; runs with `T0 = inf` were not affected.
  (`src/EOMs/time_steps.cpp`)
- **CPU-only builds write the same history files as CUDA builds.**  `energy.txt`,
  `rvec.txt`, `qk0.txt` (and `step_metrics.txt` with `-D true`) were written by code in
  `io_output.cu`, which is only compiled with CUDA, so CPU-only builds never produced
  them.  The writers are now host code in `src/io/io_save_history.cpp`, compiled in
  every build; only the GPU evaluation of the energy history remains in `io_output.cu`.
  CPU-only, CUDA-on-CPU and GPU runs now write the same files, and their contents agree
  to within the tolerance.
- **Version information is recorded at build time.**  The version, Git commit, branch
  and dirty state were obtained by running `git` when the program started, i.e. from the
  current working directory rather than from the source the binary was built from;
  outside a Git checkout every run reported `v1.0.0-dev`.  They are now written into a
  generated header on every build (`cmake/dmfe_version.cmake`); builds outside a Git
  checkout report `DMFE_FALLBACK_VERSION`.  (`src/version/version_info.cpp`)
- **`-I` / `--allow-incompatible-versions` had no effect.**  The search for an earlier
  run with the same parameters skipped runs written by an incompatible version before
  the option was consulted.  With `-I true` such runs are now continued, with a warning.
  (`src/io/io_utils.cpp`)
- **No site-specific default output directory.**  Without `-o`, a binary located inside
  the home directory wrote to `/nobackups/$USER/Results/`.  This directory is now set
  with the CMake option `DMFE_HOME_OUTPUT_DIR` (empty by default), so that a default
  build always uses `Results/` below the working directory.  (`include/core/config.hpp`)
- **Crash at the end of long GPU runs.**  After the final checkpoint, the run printed
  its final values from the host copy of the state, which on the GPU is not kept up to
  date; `saveCompressedData()` had copied `t1grid` but not `QKv` to the host, so the
  index ran past the end of the host array.  Long GPU runs (e.g. 21000 steps) therefore
  ended with a segmentation fault before `correlation.txt` was closed, losing its last
  lines; in shorter runs a meaningless value was read and then overwritten.  The final
  values are now taken from the final checkpoint, or from the host after a full copy.
  (`src/simulation/simulation_runner.cpp`)
- **Step reduction after consecutive rejections.**  After a rollback the step size was
  set to half of the last interval of the restored history.  A second rejection right
  after a first therefore restarted from the same interval, so that the step never
  became smaller and each further rejection only rewound ten more slices.  The step is
  now halved from the smaller of the restored interval and the rejected step, so that
  consecutive rejections reduce it geometrically.  Isolated rejections behave as before.
  (`src/simulation/simulation_runner.cpp`)
- **CPU step controller now identical to the GPU one.**  The step
  rejection/rollback branch was compiled and gated for the GPU only, so on the CPU
  a step with `eps_RK > 2 eps` was accepted and only the next step was shortened
  by 0.9.  The CPU now rejects such a step exactly as the GPU does: it rolls back
  the last 10 steps, halves the step and switches to SSPRK(10,4).  The
  `SSPRK(10,4) -> Dormand-Prince` demotion branch tested `rk->host->init`, which the
  GPU path never assigns, so it only ever ran on the CPU; it has been removed.
  CPU trajectories therefore differ from the released ones whenever a step is
  rejected.  (`src/simulation/simulation_runner.cpp`)
- **macOS / Apple Clang build.**  Four independent blockers: `find_package(OpenMP)`
  could not locate Homebrew's libomp; `OpenMP_CXX_FLAGS` was appended without
  `separate_arguments()`, so `"-Xclang -fopenmp"` reached the compiler as a single
  argument; `<sys/sysinfo.h>` is Linux-only; and `system_clock` was subtracted from
  `high_resolution_clock` (different clocks under libc++).  In addition
  `-Wno-overriding-t-option` was renamed by Clang to `-Wno-overriding-option`, and
  `-static` (used by `DMFE_STATIC_OPENMP` / `DMFE_PORTABLE_BUILD`) cannot link on
  Darwin.  All `#include <omp.h>` are now guarded by `_OPENMP`, so a build without
  OpenMP degrades to serial instead of failing.

- **`src/core/host_device_utils.{hpp,cu}` removed: a complete duplicate of live
  code.**  The three host helpers it declared (`Product`, `scaleVec`,
  `printVectorDifference`) are defined in `src/core/host_utils.cpp`, and the five
  device helpers (`SubtractGPU`, `scalarMultiply`, `scalarMultiply_ptr`,
  `AddSubtractGPU`, `FusedUpdate`) in `src/core/device_utils.cu`, with the same
  signatures.  The `.cu` was never listed in `CMakeLists.txt`, so the duplicate
  definitions never reached the linker -- but adding it to the build would have
  reproduced exactly the silent ODR clash fixed above for `backend_dispatch.cpp`,
  and the header's declaration of `FusedUpdate` had already drifted from the real
  one.  `src/sparsify/sparsify_utils.cu`, the only file that included the header,
  used nothing from it.

### Added
- `--error-norm sum|mean|rms|max`: norm applied to the `2L`-component Runge-Kutta
  error vector.  The default `sum` is the historical (extensive) 1-norm and
  reproduces previous runs bit for bit.  Written to `params.txt` as `error_norm`;
  a saved run is resumed only if its norm matches.
- `--sparsify-interval N`: number of accepted steps between maintenance passes,
  previously hard-wired to 100000.  Written to `params.txt` as
  `sparsify_interval` and compared on resume like `sparsify_sweeps`.
  `--sparsify-sweeps 0` now really disables sparsification in CPU-only builds as
  well.
- `grid --theta-file FILE`: build all phi grids, quadrature weights and
  interpolation stencils from an arbitrary user-supplied `theta` grid.
- `tests/`: scientific regression suite: analytic `p=2` benchmark, sparsification
  invariance, equilibrium invariants of the mixed 3+4 model started at `T0 = T`
  (constant energy and Lagrange multiplier, which exercises the `beta_0` forcing and the
  thermal noise), and a guarded CPU/GPU cross-check.  The CI build (Linux, CPU-only) runs
  it on every push and pull request.  The analytic benchmark fails with the released
  code: its CPU error is 1.4e-4 against a tolerance of 1e-6, caused by the RK54
  stage-evaluation defect above.
- Long-option aliases `--bath-temperature` (= `--Gamma`) and
  `--initial-temperature` (= `--T0`), and an unambiguous `--help` text: `Gamma` is
  the bath temperature `T` of the quench, `T0` the temperature of the equilibrium
  initial condition.

### Changed
- `CITATION.cff` now identifies the project as DYNAMITE, lists the four authors,
  carries the release version and a placeholder for the Zenodo DOI.
- Prebuilt executables (`RG-Evo`, `RG-Evo-shared`, `test_interp_weights`) are no
  longer tracked; they are ignored via `.gitignore`.

### Performance
All changes in this section are marked in the source with a `[PERF]` comment and
are **bit-for-bit neutral**: the released binary (`de696f84`) and this one produce
byte-identical `correlation.txt`, `QK_compressed`, `QR_compressed` and
`t1_compressed.txt` on the GPU, and byte-identical output on a single-threaded CPU
run.  Nothing below changes an arithmetic expression, an accumulation order, a
reduction block size or a compiler flag; the changes are to how work is *issued*,
not to what is computed.  Measured on an NVIDIA H100 PCIe 80 GB (nvcc 13.1,
gcc 14.2), `-p 3 -q 4 -l 1 -T inf -L 512 -e 1e-8 -s true`:

| | released `de696f84` | this release | |
|---|---|---|---|
| start-up (grid load + init, `-m 1`) | 2.37 s | **0.75 s** | 3.2x |
| `-t 0.5` (1124 accepted steps) | 4.31 s | **2.26 s** | 1.91x |
| `-t 5` (1587 accepted steps) | 5.07 s | **2.85 s** | 1.78x |
| `-t 20` (2009 accepted steps) | 5.77 s | **3.44 s** | 1.68x |

Subtracting start-up, the time-stepping loop itself is 1.26-1.29x faster at every
history length we measured, and now spends 1.12 s of its 1.51 s (at `-t 0.5`)
inside CUDA kernels: what remains to be won on the host side is at most a quarter
of the loop.

Start-up
- **Grid files are parsed ~10x faster.**  `Grid_data/512/` holds 9.5 million
  numbers in text form (192 MB); they were read one at a time with `file >> v`,
  i.e. ~1.4 s per run inside `std::num_get`.  Each file is now read once into
  memory and scanned with `std::from_chars` (`std::strtod` where the standard
  library has no floating-point overload).  Both are required to return the value
  nearest the decimal text under round-to-nearest, which is what `num_get` did, so
  the parsed doubles are unchanged.  (`src/io/io_input.cpp`)
- **The CUDA toolkit version is stamped at configure time** instead of being
  discovered by running `nvcc --version` through `popen()` on every run.  This also
  fixes the long-standing "No CUDA" line in `params.txt` on compute nodes that have
  no `nvcc` in `PATH`.  (`CMakeLists.txt`, `src/version/version_info.cpp`)
- **The stencil weights are no longer copied back from the device on every save**
  (~192 MB over PCIe at `L = 512`).  They are read-only grid data that no kernel
  writes, so the host arrays were already authoritative.
  (`src/core/gpu_memory_utils.cu`)

Time-stepping loop (1.26-1.29x; the loop is now within ~26% of the total GPU kernel time)
- **Kernel launches per accepted step: 229 -> 147.**  Groups of independent
  element-wise or per-block launches are issued as one kernel with a
  `(blocks x n)` grid, `blockIdx.y` selecting the entry: the eight convolutions of
  `QKRstepGPU` (also 4.5x less GPU time), the two in `rstepGPU` and the six in
  `drstep2GPU`, the four `SigmaK`/`SigmaR` evaluations of `interpolateGPU` (1.9x
  less GPU time), the A1/A2 `indexVecLN3` pair, and the `computeMA` /
  `computeWeightedSum` groups of every Runge-Kutta driver, and the two binary
  searches of `interpolateGPU` (over the concatenated index space, since theta has
  L elements and phi2 has L*L).  Each entry runs the body of the single kernel
  unchanged, with the same `blockDim` and the same reduction tree, so every output
  is the same bit pattern.  Two more launches per stage disappear by fusion rather
  than batching: the `computeProduct` that followed `FusedUpdate` in `QKRstepGPU`
  is now that kernel's optional final factor, and `updateTimeScalarsKernel` -- a
  `<<<1,1>>>` launch on a second stream -- is the `i == 0` branch of
  `append4Kernel` / `replace4Kernel`, which already had `dt` in hand (this also
  removes a benign race in which both kernels stored `t1grid[idx]`).
- **Blocking one-element copies per accepted step: 12 -> 6.**  `r(t)` is written
  straight into `rvec[idx]` by the kernel that computes it instead of travelling
  device -> host -> device; `qk0` is read on the device inside `QKRstepGPU`; the
  `h -> h0` snapshots and the stage-0 broadcast are kernels instead of synchronous
  `thrust` copies; and `t1grid.back()`, which is always a double the host itself
  wrote, is mirrored on the host (`include/core/t1_cache.hpp`) and invalidated by
  every other route that can change the grid.
- **`cudaDeviceSynchronize()` per run: 24730 -> 1.**  The device-wide barriers in
  `interpolateGPU`, `QKRstepGPU` and `drstep2GPU` are replaced by the exact
  dependency edges they stood for -- `cudaEventRecord` / `cudaStreamWaitEvent`
  (`include/core/gpu_events.hpp`), or simply issuing producer and consumer on the
  same stream.  Callers that do not name a stream pool (initialization,
  sparsification, the run controller) keep the barrier, and debug builds keep it
  everywhere.
- Long-lived CUDA events replace the two-per-stage `cudaEventCreateWithFlags`
  calls, which were also never destroyed.

Measured and rejected (kept as comments in the source so they are not retried
blindly): unrolling `indexVecLN3` for the shipped depth 10; staging the stencil
weights in shared memory; storing the device weights transposed (`d`-major) for
coalescing; and raising the occupancy hint of the two interpolation kernels from
`__launch_bounds__(64, 1)` to `(64, 16)`.  All four are bit-exact and all four are
slower on an H100: these kernels are limited by the memory-level parallelism of a
single thread, and their apparently uncoalesced access is already DRAM-optimal
because the lines one load touches are the lines the next `depth-1` loads need.

### Known issue (pre-existing, not introduced here)
- The **CPU backend is not reproducible run to run** with more than one OpenMP
  thread: two identical invocations of the same binary give different
  `correlation.txt`.  With `OMP_NUM_THREADS=1` it is deterministic, and identical
  to the released code.  The GPU backend is bit-for-bit reproducible in all the
  configurations we tested.

## [1.3.3.0] - 2026-06-26 (`263b2226`)
Cubic interpolation of the `t1` grid promoted to the CPU branch.

## [1.3.2.2] - 2025-12-02 (`470d38e6`)
Index-space interpolation (`index-poly`, `index-rational`, `index-bspline`,
`index-hermite`) and the `index-spline` integration weights; `index-poly` order 9
becomes the default.  **Results reported in the article require this revision or
later.**

## [1.3.2.1] - 2025-12-01 (`1e44ea62`)
## [1.3.2.0] - 2025-12-01 (`944e8553`)
Documentation and repository rename `DMFE` -> `DYNAMITE`.

## [1.3.1.3] - 2025-10-12 (`bfb0dc36`)
## [1.3.1.2] - 2025-10-10 (`5e3b33c7`)
## [1.3.1.0] - 2025-10-10 (`d062d264`)
## [1.3.0.1] - 2025-10-07 (`1aa62fdc`)
GitHub Pages documentation.

## [1.2.0.3] - 2025-10-04 (`f007266c`)
## [1.2.0.2] - 2025-10-04 (`29b9a847`)
## [1.2.0.1] - 2025-10-04 (`4c1b56df`)
## [1.1.0.3] - 2025-09-19 (`27360a2a`)
## [1.0.5.1] - 2025-09-17 (`32f33988`)
## [1.0.5.0] - 2025-09-17 (`41be33c1`)
## [1.0.4.0] - 2025-09-16 (`a54a2d90`)
## [1.0.3.0] - 2025-09-16 (`a82ed67a`)
## [1.0.2.5] - 2025-09-16 (`0d6514a9`)
## [1.0.2.4] - 2025-09-16 (`afab27b7`)
## [1.0.2.3] - 2025-09-16 (`35d36942`)
## [1.0.1.4] - 2025-09-15 (`7bd2006a`)
## [1.0.0.0] - 2025-07-28 (`2324f649`)
Initial public version.
