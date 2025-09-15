## DMFE Overview

High-performance CUDA/C++ implementation for DMFE simulations. The code builds a reusable core library and a single executable that drives the simulation.

### Targets
- Library: `DFME-core` (static) — shared GPU/host modules
- Executable: `RG-Evo` — main entry point, placed in the project root on build

---

## Build

Requirements
- CUDA toolkit (NVCC); a modern NVIDIA GPU with a matching driver
- CMake ≥ 3.24
- C++17 compiler (clang++-14 is preferred as NVCC host compiler if available)

Quick start
- With helper script:
  - `./build.sh` (supports `--clean`)
- Manually:
  - Configure: `cmake -S . -B build`
  - Build: `cmake --build build -j` 

Config options (toggle at configure time)
- `-DUSE_HDF5=ON` — enable compile-time HDF5 linkage (requires dev packages)
- `-DUSE_HDF5_RUNTIME=ON` — enable runtime-optional HDF5 via dlopen (default ON)
- `-DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"` — set GPU arch list (override to match your GPU)
- `-DCMAKE_BUILD_TYPE=Release|RelWithDebInfo|Debug` — choose build type (default Release)

Notes
- The executable is written to the project root (`./RG-Evo`).
- When `USE_HDF5_RUNTIME=ON`, HDF5 is loaded at runtime if present; host link uses `-ldl` only.

---

## Run

After building:
- `./RG-Evo -h` — show usage
- `./RG-Evo [options]` — run a simulation

Options (from the built-in help)
- `-p INT`      Set p parameter (default: current config)
- `-q INT`      Set p2 parameter
- `-l FLOAT`    Set lambda parameter
- `-T FLOAT`    Set T0 parameter; use `inf` for infinity
- `-G FLOAT`    Set Gamma parameter
- `-m INT`      Set maximum number of loops
- `-L INT`      Set grid length/size (compiled paths expect matching grid assets)
- `-t FLOAT`    Set maximum simulation time
- `-d FLOAT`    Set minimum time step
- `-e FLOAT`    Set maximum error per step
- `-s BOOL`     Enable output saving (correlation/state/compressed data)
- `-S BOOL`     Use SERK2 fused SigmaK/SigmaR GPU kernel (default: true)
- `-D BOOL`     Enable debug mode
- `-v`          Display version information and exit
- `-c FILE`     Check version compatibility of parameter file and exit
- `-h`          Display help and exit

Examples
- Minimal run: `./RG-Evo -p 3 -q 12 -l 0.3 -T inf -G 0.0 -L 512`
- Tuning timestepping: `./RG-Evo -d 1e-5 -e 1e-10 -t 1e7`
- Version info only: `./RG-Evo -v`
- Check a parameter file: `./RG-Evo -c Results/params.txt`

Data and outputs
- Precomputed grid data lives under `Grid_data/` (e.g., `512/`, `1024/`, `2048/`).
- When saving is enabled (`-s true`), outputs (correlations, state, optional HDF5) are written by the I/O layer; see `include/io_utils.hpp` for details.

---

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
  - convolution: `convolution.hpp`
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

## Troubleshooting
- Host compiler: If NVCC warns about the host compiler, install `clang++-14` and re-configure; the build prefers it automatically when found.
- GPU architectures: If your GPU is older/newer than the defaults, set `-DCMAKE_CUDA_ARCHITECTURES` accordingly (e.g., `75` for Turing, `90` for Hopper).
- Clean rebuild: Use `./build.sh --clean` or delete the `build/` directory and re-run CMake.

