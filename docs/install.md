# <img class="icon icon-lg icon-primary" src="/DMFE/assets/icons/cpu.svg" alt="Install icon"/> Install & Build

DMFE builds with CMake for CPU and optionally CUDA GPUs.

## Prerequisites

- CMake >= 3.24
- C++17 compiler (GCC or Clang) with OpenMP
- Optional: CUDA Toolkit (enables GPU build)
- Optional: HDF5 development libraries (compile-time HDF5)
- Docs build: Doxygen (Graphviz optional for diagrams)

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

### No sudo / user-space installs

- Python tools (mkdocs, doxybook2): use `pip install --user ...` or a local environment (e.g., micromamba/conda).
- Doxygen: use a prebuilt binary or container if your system lacks packages.
- Graphviz is optional; API pages generate without diagrams when not present.

---

License: Apache-2.0. See the repository `LICENSE` file. For citation instructions, see Reference → Cite.
