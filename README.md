# DMFE

DMFE is a CUDA/C++ solver for long-time, non‑stationary dynamics governed by dynamical mean‑field equations. It implements a numerical renormalization scheme based on two‑dimensional interpolation of correlation and response functions, reducing the cost of aging dynamics from cubic to sublinear in simulated time. The code was introduced in “Numerical renormalization of glassy dynamics” (Lang, Sachdev, Diehl; arXiv:2504.06849), where it reaches time scales orders of magnitude beyond previous methods and resolves a finite‑temperature transition between strongly and weakly ergodicity‑broken glasses in the spherical mixed p‑spin model. While validated on a glassy system, the approach applies broadly to models with overdamped excitations.

Key features:
- GPU‑accelerated kernels with a CPU fallback
- Non‑equilibrium quench dynamics with automatic checkpoint/resume
- Parameter‑organized outputs (HDF5 or binary) and lightweight text logs
- Ready‑to‑use interpolation grids under `Grid_data/<L>/`

## Build

From the project root (this directory):

```bash
./build.sh
```

Or manually:

```bash
cmake -S . -B build
cmake --build build -j $(nproc)
```

If you previously ran cmake from a parent directory accidentally (e.g. from $HOME), remove the stray build directory you created and re-run from here.

## Run

After build the executable is placed in the project root:

```bash
./RG-Evo
# If enabled, a shared-runtime variant is also produced:
./RG-Evo-shared
```

Show help and defaults:

```bash
./RG-Evo -h
```

## Build options

Pass options to CMake at configure time (either via `./build.sh ...` or directly with `cmake -S . -B build ...`). Defaults shown in parentheses.

- CMAKE_BUILD_TYPE (Release) — Standard CMake build type: Release, Debug, RelWithDebInfo, MinSizeRel.
- CMAKE_CUDA_ARCHITECTURES (80;86;89;90) — Space- or semicolon-separated SM list. Example: `-DCMAKE_CUDA_ARCHITECTURES="80;90"`.
- DMFE_DEBUG (OFF) — Enables CUDA device debug flags (-G, lineinfo), disables fast-math, sets THRUST_DEBUG=1.
- DMFE_NATIVE (ON) — Adds `-march=native` for host compilation when using GCC/Clang.
- DMFE_PORTABLE_BUILD (OFF) — Portable/cluster helper. Forces `DMFE_NATIVE=OFF` and uses shared CUDA runtime (see below).
- DMFE_STATIC_CUDART (ON) — Link CUDA runtime statically; helps run on systems without a local CUDA toolkit. Set `OFF` to use shared cudart.
- DMFE_BUILD_SHARED_VARIANT (ON) — Also build `RG-Evo-shared` with shared CUDA runtime.
- USE_HDF5 (OFF) — Compile-time HDF5 linkage. Requires dev packages (e.g., `libhdf5-dev`). Defines `USE_HDF5` and links `HDF5::HDF5`.
- USE_HDF5_RUNTIME (ON) — Runtime-optional HDF5 via dlopen. Adds `src/io/h5_runtime.cpp`, defines `H5_RUNTIME_OPTIONAL=1`, links `-ldl`. Note: Many distro builds hide the `H5T_*` native type globals; for best compatibility, ensure the HDF5 High-Level runtime library is present (e.g. `libhdf5_hl.so` / `libhdf5_serial_hl.so`). If the loader prints that it loaded `libhdf5*.so` but not the HL library and HDF5 writes fail, the program will fall back to `data.bin`.
- DMFE_PREFER_CLANG_HOST (OFF) — Prefer `clang++-14` as CUDA host compiler if available.

Examples:

```bash
# Release build with explicit arch list
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;86;90"

# Debug build with device debug and thrust checks
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DDMFE_DEBUG=ON

# Portable build for heterogeneous clusters (no -march=native, shared cudart)
cmake -S . -B build -DDMFE_PORTABLE_BUILD=ON -DCMAKE_CUDA_ARCHITECTURES=80

# Enable compile-time HDF5 (system must provide HDF5 dev libs)
cmake -S . -B build -DUSE_HDF5=ON

# Disable runtime-optional HDF5 wrapper
cmake -S . -B build -DUSE_HDF5_RUNTIME=OFF

# Build only static cudart variant and skip shared one
cmake -S . -B build -DDMFE_BUILD_SHARED_VARIANT=OFF -DDMFE_STATIC_CUDART=ON
```

Toolchain hints:

- The helper `build.sh` tries NVHPC (`nvc++`) first (via environment modules if available), otherwise selects a GCC/Clang version compatible with your CUDA.
- To override compilers manually, pass standard CMake variables, e.g.:

```bash
cmake -S . -B build \
	-DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 \
	-DCMAKE_CUDA_HOST_COMPILER=g++-12
```

## Clean build

```bash
./build.sh --clean        # cleans ./build
# or
./build.sh build --clean  # equivalent
```

## Usage

- Typical run:

```bash
./RG-Evo -m 120000 -D false -q 4 -l 0.5 -L 512
```

- Common flags (see `-h` for full list and defaults):
	- `-p INT` primary parameter p
	- `-q INT` secondary parameter p2
	- `-l, --lambda FLOAT` coupling lambda
	- `-T, --T0 FLOAT|inf` temperature scale (use `inf` for infinity)
	- `-G, --Gamma FLOAT` Gamma
	- `-L INT` grid length L (must match available data, e.g. 512/1024/2048)
	- `-m INT` max number of loops
	- `-t FLOAT` max simulation time
	- `-d FLOAT` minimum time step
	- `-e, --error FLOAT` max error per step
	- `-o, --out-dir DIR` directory to write all outputs into (overrides defaults)
	- `-s BOOL` save outputs (default true; pass `false` to disable)
	- `-S, --serk2 BOOL` use SERK2 method (default true)
	- `-a, --aggressive-sparsify BOOL` enable aggressive sparsification (default true)
	- `-D BOOL` debug messages (default true)
	- `-g, --gpu BOOL` enable GPU acceleration (default true)
	- `-A, --async-export BOOL` enable asynchronous data export (default true)
	- `-v` print version info and exit
	- `-c, --check FILE` check version compatibility of a params file and exit

GPU/CPU: By default, the program attempts to use GPU acceleration if compatible hardware is detected (`--gpu true`). You can explicitly disable GPU acceleration with `--gpu false` to force CPU-only execution. The program will automatically fall back to CPU if GPU is disabled or if no compatible GPU hardware is found.

## Data Export Modes

The program supports two data export modes controlled by the `--async-export` flag:

- **Asynchronous export** (`--async-export true`, default): Data saving operations run in background threads, allowing the simulation to continue without waiting for I/O operations to complete. This improves performance for long-running simulations but carries a small risk of data loss if the program terminates unexpectedly before background saves complete.

- **Synchronous export** (`--async-export false`): All data saving operations complete before the simulation continues. This ensures data integrity but may slow down the simulation during save operations, especially for large datasets.

For most use cases, asynchronous export is recommended as it provides better performance while the built-in synchronization ensures data is properly saved at program termination. Use synchronous export if you need guaranteed data integrity at every save point or are experiencing issues with missing files.

## Inputs (required data)

Interpolation grids are loaded at startup from `Grid_data/<L>/`, where `<L>` is the grid length you pass via `-L` (defaults to 512). The directory must contain:

- `theta.dat`, `phi1.dat`, `phi2.dat`, `int.dat`
- `posA1y.dat`, `posA2y.dat`, `posB2y.dat`
- `indsA1y.dat`, `indsA2y.dat`, `indsB2y.dat`
- `weightsA1y.dat`, `weightsA2y.dat`, `weightsB2y.dat`

Example: `Grid_data/512/...` for `-L 512`.

## Outputs (where data is saved)

At startup, an output root directory is selected:

- If you pass `-o /path/to/dir` (or `--out-dir /path/to/dir`), all results are written there. The path is created if needed.
- Otherwise, if the executable resides under your `$HOME`, results are written under `config.outputDir` (default in code: `/nobackups/jlang/Results/`).
- Else results go to `config.resultsDir` (default `Results/`, relative to the working dir).

Both defaults live in `include/config.hpp`. The CLI `--out-dir` provides a runtime override without rebuilding.

Example:

```bash
./RG-Evo -L 512 -l 0.5 -o /scratch/$USER/dmfe_runs
```

Within the selected root, results are organized by parameters:

```
<root>/p=<p>_p2=<p2>_lambda=<lambda>_T0=<inf|value>_G=<Gamma>_len=<L>/
```

Files produced (depending on HDF5 availability and flags):

- `data.h5` — main state (preferred when HDF5 is available)
	- Datasets: `QKv`, `QRv`, `dQKv`, `dQRv`, `t1grid`, `rvec`, `drvec`
	- Attributes: `time`, `iteration`, `len`, `delta`, `delta_t`, `T0`, `lambda`, `p`, `p2`, `energy`, plus build/version metadata
- `data.bin` — binary fallback when HDF5 is not available
- `params.txt` — human-readable parameters, runtime stats, environment, version info
- `correlation.txt` — tab-separated time, energy, QK[0] samples during run
- `energy.txt`, `rvec.txt`, `qk0.txt` — histories derived at save time
- `QK_compressed`, `QR_compressed` — compact binary snapshots of selected arrays

Saving occurs periodically and at the end when `-s true` (default). Disable with `-s false`.

## Resume/continue runs

On startup, the code looks for an existing checkpoint in the parameter directory (`data.h5` or `data.bin`) and resumes automatically if parameters match. A compatibility check against `params.txt` is performed when present, including version compatibility: versions are considered compatible if at least the first two positions of the version number agree (e.g., v1.2.3.4 is compatible with v1.2.5.6). If compatible data is found, the simulation will save its output into the existing directory. To inspect a params file without running:

```bash
./RG-Evo -c /path/to/Results/.../params.txt
```
