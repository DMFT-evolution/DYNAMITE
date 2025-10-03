# Install & Build

DMFE builds with CMake for CPU and optionally CUDA GPUs.

## Prerequisites

- CMake >= 3.24
- A C++17 compiler (GCC or Clang). OpenMP recommended.
- Optional: CUDA Toolkit (for GPU build)
- Optional: HDF5 dev libs (for compile-time HDF5 linkage)

## Build from source

Recommended:

```bash
./build.sh
```

Manual:

```bash
cmake -S . -B build
cmake --build build -j $(nproc)
```

Executables: `./RG-Evo` (and optionally `./RG-Evo-shared`).

## Options

- `-DDMFE_DEBUG=ON` — device debug flags, thrust checks
- `-DDMFE_PORTABLE_BUILD=ON` — no `-march=native`, shared cudart
- `-DCMAKE_CUDA_ARCHITECTURES="80;86;90"` — SM list
- `-DUSE_HDF5=ON` — compile-time HDF5
- `-DUSE_HDF5_RUNTIME=OFF` — disable runtime-optional HDF5 wrapper

See README Build options for the full list and examples.

## CUDA notes

- If CUDA is installed and a compiler is detected, CUDA is enabled by default.
- For mixed GCC/CUDA versions, prefer `clang++-14` as host for nvcc or use GCC 11/12.
- For clusters, use `-DDMFE_PORTABLE_BUILD=ON` and set a single SM: `-DCMAKE_CUDA_ARCHITECTURES=80`.

## Troubleshooting

- CMake can’t find CUDA: set `-DDMFE_WITH_CUDA=OFF` to build CPU-only; verify `nvcc --version`.
- Linker errors with libgomp/libomp: set `-DDMFE_STATIC_OPENMP=OFF` or install matching OpenMP.
- Runtime: missing grid data → ensure `Grid_data/<L>/` exists for your `-L`.
