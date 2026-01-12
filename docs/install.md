# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/cpu.svg" alt="Install icon"/> Install & Build

DYNAMITE builds with CMake for CPU and optionally CUDA GPUs.

## Prerequisites

- CMake >= 3.24
- C++17 compiler (GCC or Clang) with OpenMP
- Optional: CUDA Toolkit (enables GPU build)
- Optional: HDF5 development libraries (only needed if you want compile-time linked HDF5)
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

## Examples

CPU-only (portable baseline):

```bash
./build.sh --cuda=off
```

CUDA build (GPU enabled if a CUDA toolchain is found):

```bash
./build.sh --cuda=on
```

Manual equivalents:

```bash
cmake -S . -B build -DDMFE_WITH_CUDA=OFF
cmake --build build -j $(nproc)
```

## Options

- `-DDMFE_WITH_CUDA=ON|OFF` — enable/disable CUDA support.
	- Note: `./build.sh` also accepts `--cuda=auto|on|off` and will pass the appropriate CMake option.
- `-DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"` — CUDA SM list (only relevant when CUDA is enabled).
- `-DDMFE_DEBUG=ON` — debug build flags (host debug; and when CUDA is enabled, device debug + extra checks).

Host portability/performance:

- `-DDMFE_NATIVE=ON|OFF` — enable/disable `-march=native` on the host compiler.
- `-DDMFE_PORTABLE_BUILD=ON` — convenience switch for cluster/portable builds (disables `-march=native`, prefers portable linkage settings).
- `-DDMFE_STATIC_OPENMP=ON|OFF` — attempt static OpenMP linkage for portability (may require toolchain support).

### HDF5 output (compile-time vs runtime)

DYNAMITE supports three distinct situations; separating them helps avoid confusion:

1) **Compile-time linked HDF5** (requires HDF5 dev headers/libs at build time)

- Enable with `-DUSE_HDF5=ON`.
- Result: the binary is linked against HDF5, so `data.h5` is available regardless of runtime library search paths (assuming the system dynamic loader can find the linked libs).

2) **Runtime-optional HDF5 (dlopen)** (does *not* require HDF5 at build time)

- Enable with `-DUSE_HDF5_RUNTIME=ON` (this is the default in the CMake configuration).
- Result: at runtime, DYNAMITE tries to load HDF5 via `dlopen` and write `data.h5`.
	- If HDF5 libraries are found and load successfully: you get `data.h5`.
	- If HDF5 libraries are missing / fail to load / writing fails: DYNAMITE automatically falls back to the fully supported binary `data.bin`.

3) **No HDF5 at all**

- Configure with `-DUSE_HDF5=OFF -DUSE_HDF5_RUNTIME=OFF`.
- Result: the run will always write `data.bin` (plus text summaries and `params.txt`).

CUDA runtime linkage (CUDA builds only):

- `-DDMFE_STATIC_CUDART=ON|OFF` — link CUDA runtime statically (ON) or dynamically (OFF).
- `-DDMFE_BUILD_SHARED_VARIANT=ON|OFF` — also build a shared-runtime CUDA variant.

See README Build options for the full list and examples.

## CUDA notes

- If CUDA is installed and a compiler is detected, CUDA is enabled by default.
- For mixed GCC/CUDA versions, prefer `clang++-14` as host for nvcc or use GCC 11/12.
- For clusters, use `-DDMFE_PORTABLE_BUILD=ON` and set a single SM: `-DCMAKE_CUDA_ARCHITECTURES=80`.

## Troubleshooting

- CMake can’t find CUDA: build CPU-only with `-DDMFE_WITH_CUDA=OFF` (or use `./build.sh --cuda=off`); verify `nvcc --version`.
- Linker errors with libgomp/libomp: set `-DDMFE_STATIC_OPENMP=OFF` or install matching OpenMP.
- Runtime: missing grid data → ensure `Grid_data/<L>/` exists for your `-L`.

### HDF5 not found at runtime

If you built with runtime-optional HDF5 enabled (`USE_HDF5_RUNTIME=ON`), the program prints which HDF5 libraries were found (if any). If none are found (or HDF5 writing fails), DYNAMITE still saves the full state via `data.bin`.

If you built with compile-time HDF5 (`USE_HDF5=ON`) and see HDF5 load/link errors at runtime, it usually indicates a system library path issue (e.g. missing `libhdf5.so` in the dynamic loader path). In that case either fix your environment/library paths or switch to the runtime-optional mode.

### No sudo / user-space installs

- Python tools (mkdocs, doxybook2): use `pip install --user ...` or a local environment (e.g., micromamba/conda).
- Doxygen: use a prebuilt binary or container if your system lacks packages.
- Graphviz is optional; API pages generate without diagrams when not present.

---

License: Apache-2.0. See the repository `LICENSE` file. For citation instructions, see Reference → Cite.
