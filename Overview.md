## DMFE Overview

High-performance CUDA/C++ implementation for DMFE simulations. The code builds a reusable core library and a single executable that drives the simulation.

### Targets
- Library: `DFME-core` (static) — shared GPU/host modules
- Executable: `RG-Evo` — main entry point, placed in the project root on build

For build, run, inputs/outputs, and full CLI reference, see `README.md`.

## Module Map

Public headers are under `include/`, sources under `src/`:
- core: device/host utils, config/initialization, I/O helpers
- interpolation: grid and vector interpolation kernels and dispatchers
- convolution: convolution kernels
- math: math helpers and sigma kernels/wrappers
- EOMs: time-stepping and Runge–Kutta implementations
- search: search/utility kernels
- sparsify: sparsification helpers
- simulation: simulation control and runner
- version: build/version info and compatibility helpers

---

## Folder structure

- Root
  - `CMakeLists.txt` — build configuration (C++/CUDA, targets DFME-core and RG-Evo)
  - `main.cu` — program entry (version print, CLI parse, init, run)
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
  - `io/` — optional runtime HDF5 helpers
  - `version/` — version info and compatibility

---

## Function locations (quick reference)

- Entry/CLI
  - `main.cu`: prints version, calls `parseCommandLineArguments`, then `init` and `runSimulation`
  - `parseCommandLineArguments` — `src/core/config.cu`, declared in `include/config.hpp`
- Initialization/State
  - `init` — `src/core/initialization.cu`, declared in `include/initialization.hpp`
  - `rollbackState` — `src/simulation/simulation_control.cu`, declared in `include/simulation_control.hpp`
  - Simulation data structures — `include/simulation_data.hpp`
- Simulation loop
  - `runSimulation` — `src/simulation/simulation_runner.cu`, declared in `include/simulation_runner.hpp`
- I/O
  - `fileExists`, `saveHistory`, helpers — `src/core/io_utils.cu`, declared in `include/io_utils.hpp`
  - Optional HDF5 runtime helpers — `src/io/h5_runtime.cpp`, interface in `include/io/h5_runtime.hpp`
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
``` 

