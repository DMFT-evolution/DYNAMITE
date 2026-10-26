# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/cpu.svg" alt="Install icon"/> Install

DYNAMITE is built from source with CMake. A CPU-only build needs nothing beyond a C++ compiler; CUDA is optional.

## Requirements

- CMake ≥ 3.24
- A C++17 compiler (GCC or Clang)
- OpenMP (optional, strongly recommended): without it the CPU code runs on a single thread
- CUDA toolkit (optional): enables the GPU backend
- HDF5 libraries (optional): only used to write the checkpoint as `data.h5` instead of `data.bin`

The CPU-only build is tested on Linux and macOS. On macOS, OpenMP comes from Homebrew (`brew install libomp`), which CMake finds automatically.

## Build

From the repository root:

```bash
./build.sh               # CUDA if a CUDA compiler is found, otherwise CPU-only
./build.sh --cuda=off    # force a CPU-only build
./build.sh --cuda=on     # require CUDA
```

or with CMake directly:

```bash
cmake -S . -B build                       # add -DDMFE_WITH_CUDA=OFF for CPU-only
cmake --build build -j "$(getconf _NPROCESSORS_ONLN)"
```

The executable `RG-Evo` is placed in the repository root. A CUDA build also produces `RG-Evo-shared`, which links the CUDA runtime dynamically; both behave identically. Check the build with

```bash
./RG-Evo -v      # version, compiler and CUDA version
./RG-Evo -h      # all options and their defaults
```

A CUDA build also runs on machines without a GPU: it then uses the CPU backend.

## CMake options

| Option | Default | Effect |
|---|---|---|
| `DMFE_WITH_CUDA` | `ON` | Build the GPU backend if a CUDA compiler is found. |
| `CMAKE_CUDA_ARCHITECTURES` | `80;86;89;90` | GPU architectures to compile for, e.g. `80` (A100) or `90` (H100). A single value shortens the build. |
| `DMFE_NATIVE` | `ON` | Compile host code with `-march=native`. Turn off if the binary must run on other CPUs. |
| `DMFE_PORTABLE_BUILD` | `OFF` | Settings for heterogeneous clusters: no `-march=native`, portable linking. See [Cluster build](tutorials/cluster-portable-build.md). |
| `DMFE_STATIC_OPENMP` | `OFF` | Link OpenMP statically. |
| `DMFE_STATIC_CUDART` | `ON` | Link the CUDA runtime statically (CUDA builds only). |
| `DMFE_BUILD_SHARED_VARIANT` | `ON` | Also build `RG-Evo-shared` (CUDA builds only). |
| `USE_HDF5_RUNTIME` | `ON` | Load the HDF5 library at run time if it is installed; otherwise write `data.bin`. |
| `USE_HDF5` | `OFF` | Link HDF5 at build time instead (requires the HDF5 development files). |
| `DMFE_DEBUG` | `OFF` | Debug build with device-side checks. Slow. |
| `DMFE_HOME_OUTPUT_DIR` | empty | Output directory used when the executable lies inside your home directory and no `-o` is given, e.g. `/nobackups/{USER}/Results/` (`{USER}` is replaced by the user name). Empty: `Results/` below the working directory, as always. |
| `DMFE_FALLBACK_VERSION` | `1.4.0.0` | Version reported by a build from a source tree that is not a Git checkout. |

With `USE_HDF5=OFF` and `USE_HDF5_RUNTIME=OFF` the checkpoint is always written as `data.bin`. Both formats contain the same data and both can be used to continue a run.

## Troubleshooting

- **CMake does not find CUDA** (the configuration reports a CPU-only build). Pass the compiler explicitly, `-DCMAKE_CUDA_COMPILER=/path/to/nvcc`, or build CPU-only with `./build.sh --cuda=off`. After changing compilers, delete the `build/` directory, so that CMake does not reuse old settings.
- **`cudaErrorUnsupportedPtxVersion` at run time.** The NVIDIA driver is older than the CUDA toolkit used for the build and cannot compile the code shipped for other GPU generations. Configure with the architecture of your GPU given explicitly, e.g. `-DCMAKE_CUDA_ARCHITECTURES=89` for an RTX 4090, or update the driver.
- **CUDA and GCC versions do not match.** Older CUDA toolkits do not accept recent GCC versions. Use GCC 11 or 12 as host compiler, or configure with `-DDMFE_PREFER_CLANG_HOST=ON`.
- **Linker errors involving `libgomp` or `libomp`.** Keep `DMFE_STATIC_OPENMP=OFF` and make sure that the compiler and the OpenMP library belong together.
- **The program reports that it cannot load HDF5.** This is harmless: the checkpoint is written as `data.bin` instead.
- **Missing grid files.** If `Grid_data/<L>/` does not exist, the program generates it on the first run. See [Generate grids](tutorials/generate-grids.md).
