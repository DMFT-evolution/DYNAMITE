## DYNAMITE Overview

High-performance CUDA/C++ implementation for DMFE simulations. The code builds a reusable core library and a single executable that drives the simulation.

### Targets
- Library: `DFME-core` (static) — shared GPU/host modules
- Executable: `RG-Evo` — main entry point, placed in the project root on build

For build, run, inputs/outputs, and full CLI reference, see `README.md`.

I/O at a glance:
- Preferred format: `data.h5` (HDF5), loaded via a runtime-optional wrapper; automatic fallback to `data.bin` if HDF5 is unavailable at runtime.
- Separate artifacts: `params.txt`, histories (`rvec.txt`, `energy.txt`, `qk0.txt`), and compressed snapshots (`QK_compressed`, `QR_compressed`, `t1_compressed.txt`).
- TUI/save telemetry: progress windows for main file [0.10..0.50], params [0.50..0.65], histories [0.65..0.80], compressed [0.80..0.90], with a concise "Save started"/"Save finished: <dir>" message pair.

## Module Map

Public headers are under `include/`, sources under `src/`:
- core: device/host utils, config/initialization, I/O helpers and save telemetry
- interpolation: grid and vector interpolation kernels and dispatchers
- convolution: convolution kernels
- math: math helpers and sigma kernels/wrappers
- EOMs: time-stepping and Runge–Kutta implementations
- search: search/utility kernels
- sparsify: sparsification helpers
- simulation: simulation control and runner (async/sync save orchestration)
- version: build/version info and compatibility helpers
- io: modular writers (HDF5 or binary), params/history/compressed outputs, runtime HDF5 loader

---

## Folder structure

- Root
  - `CMakeLists.txt` — build configuration (C++/CUDA, targets DFME-core and RG-Evo)
  - `main.cpp` — program entry (version print, CLI parse, init, run)
  - `build/` — CMake build artifacts (generated)
  - `Grid_data/` — precomputed grid assets (512/, 1024/, 2048/)
  - `Overview.md`, `README.md`, `build.sh`
- `include/` — public headers for all modules
  - core APIs: `config.hpp`, `globals.hpp`, `io_utils.hpp`, `gpu_memory_utils.hpp`, `device_utils.cuh`, `host_device_utils.hpp`, `host_utils.hpp`
  - math: `math_ops.hpp`, `math_sigma.hpp`
  - interpolation: `index_mat.hpp`, `index_vec.hpp`, `interpolation_core.hpp`
  - EOMs: `time_steps.hpp`, `runge_kutta.hpp`, `rk_data.hpp`
  - search: `search_utils.hpp`
  - simulation: `simulation_control.hpp`, `simulation_runner.hpp`, `simulation_data.hpp`
  - convolution: `convolution/convolution.hpp`
  - versioning: `version_info.hpp`, `version_compat.hpp`
  - io runtime: `io/h5_runtime.hpp`
- `src/` — implementations by module
  - `core/` — config, initialization, IO, device/host utils, GPU memory
  - `interpolation/` — index_mat, index_vec, dispatchers
  - `convolution/` — convolution kernels
  - `math/` — math_ops and sigma kernels/wrappers
  - `EOMs/` — time_steps and runge_kutta
  - `search/` — CPU/GPU search utilities
  - `simulation/` — runner and control logic
  - `io/` — modular writers and helpers (runtime-optional HDF5 loader, params/history/compressed writers)
  - `version/` — version info and compatibility

---

  ## Grid generation (theta/phi/pos and metadata)

  The executable provides a `grid` subcommand to generate the non‑equidistant θ/ϕ grids and the interpolation metadata consumed at runtime. Outputs are written under `Grid_data/<L>/`.

  Quick example:

  ```
  ./RG-Evo grid --len 512 --Tmax 100000 --dir 512 \
                --interp-method poly --interp-order 9
  # Optional smooth index remapping:
  ./RG-Evo grid -L 512 --alpha 0.25 --delta 0.5 --dir 512-custom
  ```

  Key flags:
  - `--len L` grid size (512/1024/2048 supported by default); `--Tmax X` sets long‑time scale for θ.
  - `--interp-method {poly|rational|bspline}` and `--interp-order n` choose the interpolation used to precompute weights.
  - `--fh-stencil m` (rational only) sets the Floater–Hormann window size (m ≥ n+1). Default m=n+1 (equivalent to `poly`); wider m (e.g., n+3..n+7) blends overlapping local stencils for additional robustness on irregular grids while keeping locality.
  - `--spline-order s` sets the B‑spline degree for `int.dat` (default 5).
  - `--alpha X` in [0,1] and `--delta X ≥ 0` apply an optional smooth non‑linear index remapping for θ; defaults are 0 (paper‑exact grid). Values are recorded in `grid_params.txt`.

  Defaults if omitted: `--Tmax 100000`, `--interp-method poly`, `--interp-order 9`, `--spline-order 5`, and for `rational` the default `--fh-stencil` is `n+1`.

  Artifacts written: `theta.dat`, `phi1.dat`, `phi2.dat`, `int.dat`, `posA1y.dat`, `posA2y.dat`, `posB2y.dat`, and interpolation indices/weights for A1/A2/B2. The file `grid_params.txt` captures provenance, including `alpha`/`delta` when used.

  Folder layout under `Grid_data/<L>/` (brief):
  - Grids: `theta.dat` (N), `phi1.dat` (N×N), `phi2.dat` (N×N)
  - Integration: `int.dat` (N) — open‑clamped B‑spline quadrature of degree `s` (default 5)
  - Positions: `posA1y.dat`, `posA2y.dat`, `posB2y.dat`
  - Interp metadata: `indsA1y.dat`, `weightsA1y.dat`, `indsA2y.dat`, `weightsA2y.dat`, `indsB2y.dat`, `weightsB2y.dat`

  Validation mode: add `--validate` (or `-V`) to recompute θ/ϕ/int and compare against the saved files in `Grid_data/<L>/` without writing outputs; returns non‑zero on mismatch.

  Note: at simulation startup, grids for `-L` are auto‑provisioned if missing (symlink reuse or on‑the‑fly generation). See README “Automatic grid provisioning at startup”.

  ---

## Function locations (quick reference)

- Entry/CLI
  - `main.cpp`: prints version, calls `parseCommandLineArguments`, then `init` and `runSimulation`
  - `parseCommandLineArguments` — `src/core/config.cu`, declared in `include/config.hpp`
- Initialization/State
  - `init` — `src/core/initialization.cu`, declared in `include/initialization.hpp`
  - `rollbackState` — `src/simulation/simulation_control.cu`, declared in `include/simulation_control.hpp`
  - Simulation data structures — `include/simulation_data.hpp`
- Simulation loop
  - `runSimulation` — `src/simulation/simulation_runner.cu`, declared in `include/simulation_runner.hpp`
// I/O
  - Utilities & telemetry — `src/io/io_utils.cpp`, declared in `include/io/io_utils.hpp`
  - HDF5 runtime loader — `src/io/h5_runtime.cpp`, interface in `include/io/h5_runtime.hpp`
  - Writers — `src/io/io_save_hdf5.cpp`, `src/io/io_save_binary.cpp`, `src/io/io_save_params.cpp`, `src/io/io_save_compressed.cpp`, orchestrated by `src/io/io_save_driver.cpp`
- Interpolation
  - Matrix: `indexMatAll` (CPU) — `src/interpolation/index_mat.cu`; `indexMatAllGPU` (GPU) — same file; API in `include/index_mat.hpp`
  - Vector: `indexVecLN3`, `indexVecN`, `indexVecR2` (CPU) and `indexVec*GPU` (GPU) — `src/interpolation/index_vec.cu`; API in `include/index_vec.hpp`
  - High-level orchestration — `src/interpolation/interpolation_core.cu`; API in `include/interpolation_core.hpp`
- Math/Sigma
  - `computeSigmaKandRKernel` (GPU kernel) — `src/math/math_sigma.cu`; decl in `include/math_sigma.hpp`
  - `SigmaK*`, `SigmaR*` host/GPU wrappers — `src/math/math_sigma.cu`; decls in `include/math_sigma.hpp`
  - Misc math helpers — `include/math_ops.hpp`
- Device/Host utils
  - `SubtractGPU`, `scalarMultiply`, `AddSubtractGPU`, `FusedUpdate` — `src/core/device_utils.cu`; API in `include/device_utils.cuh` and `include/host_device_utils.hpp`
- Convolution
  - Host/GPU convolution routines — `src/convolution/convolution.cu`; API in `include/convolution.hpp`
- EOMs (time integration)
  - Time-stepping kernels/launchers — `src/EOMs/time_steps.cu`; API in `include/time_steps.hpp`
  - Runge–Kutta variants — `src/EOMs/runge_kutta.cu`; API in `include/runge_kutta.hpp`
- Search utilities
  - CPU/GPU search and kernels — `src/search/search_utils.cu`; API in `include/search_utils.hpp`
- Versioning
  - Version info and string — `src/version/version_info.cu`; API in `include/version_info.hpp`
  - Compatibility analysis/checks — `src/version/version_compat.cu`; API in `include/version_compat.hpp`

---

## Test utility: interpolation weights

- Executable: `test_interp_weights` (built by CMake; output placed in the project root).
- Source: `src/grid/test_interpolation_weights.cpp` (standalone; no DFME‑core linkage).
- What it does: generates a θ grid (len=512, `Tmax=1e5`) and verifies that
  - polynomial (`poly`),
  - rational (`rational`, with default stencil),
  - rational with wider window (`m=n+5`), and
  - B‑spline (global) weights
  reproduce monomials up to degree `n` at 10 query points in [0,1]. It prints the max abs error per variant and exits non‑zero on failure; on success it ends with “All variants PASSED”.

Run after a normal build from the repo root:

```
./test_interp_weights
```
``` 

