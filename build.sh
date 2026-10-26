#!/usr/bin/env bash
set -euo pipefail

# Usage: ./build.sh [name-or-path] [--clean] [--cuda=auto|on|off]
# By default, build directories are created under ./build
# Examples:
#   ./build.sh                -> ./build (CUDA auto-detected)
#   ./build.sh --cuda=off     -> ./build (CPU-only)
#   ./build.sh --cuda=on      -> ./build (CUDA required)
#   ./build.sh nvhpc          -> ./build/nvhpc
#   ./build.sh build-foo      -> ./build/build-foo
#   ./build.sh build/foo      -> ./build/foo
#   ./build.sh /tmp/dmfe-bld  -> /tmp/dmfe-bld (absolute path respected)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
RAW_DIR=""
ROOT_BUILD_DIR="build"

# Parse arguments: optional build dir + flags
CLEAN=0
CUDA_MODE="auto"  # auto, on, or off
# Optional: honor CMAKE_BUILD_TYPE if exported; default to Release
BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
parse_options=1
for arg in "$@"; do
  if [[ $parse_options -eq 1 ]]; then
    case "$arg" in
      --clean)
        CLEAN=1; continue ;;
      --cuda=auto|--cuda=on|--cuda=off)
        CUDA_MODE="${arg#--cuda=}"; continue ;;
      --)
        parse_options=0; continue ;;
      -*)
        echo "[error] Unknown option: $arg" >&2
        exit 1 ;;
    esac
  fi
  if [[ -n "$RAW_DIR" ]]; then
    echo "[error] Only one build directory may be specified" >&2
    exit 1
  fi
  RAW_DIR="$arg"
done

if [[ -z "${RAW_DIR}" || "${RAW_DIR}" == "build" ]]; then
  BDIR="${ROOT_BUILD_DIR}"
elif [[ "${RAW_DIR}" == /* ]]; then
  BDIR="${RAW_DIR}"
elif [[ "${RAW_DIR}" == build/* ]]; then
  BDIR="${RAW_DIR}"
else
  BDIR="${ROOT_BUILD_DIR}/${RAW_DIR}"
fi

# Resolve directory aliases using portable shell builtins (also works on macOS).
if [[ "$BDIR" != /* ]]; then
  BDIR="${SCRIPT_DIR}/${BDIR}"
fi
mkdir -p -- "$BDIR"
BDIR=$(cd -- "$BDIR" && pwd -P)
# Never let --clean or cache recovery remove the source tree or an ancestor.
if [[ "$BDIR" == / || "$SCRIPT_DIR" == "$BDIR" || "$SCRIPT_DIR" == "$BDIR/"* ]]; then
  echo "[error] Build directory must not contain the source directory: $BDIR" >&2
  exit 1
fi

same_directory() {
  [[ -d "$1" && -d "$2" ]] && [[ "$1" -ef "$2" ]]
}

# Both source and build locations are recorded in the cache. Compare directory
# identities so symlink aliases do not discard a valid cache and its settings.
if [[ $CLEAN -eq 0 && -f "$BDIR/CMakeCache.txt" ]]; then
  cached_source=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BDIR/CMakeCache.txt")
  cached_build=$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' "$BDIR/CMakeCache.txt")
  if ! same_directory "$cached_source" "$SCRIPT_DIR" || ! same_directory "$cached_build" "$BDIR"; then
    echo "[configure] Source or build directory moved; clearing stale CMake configure state" >&2
    echo "[configure] Cached custom options will reset; reapply any site-specific CMake settings" >&2
    rm -rf -- "$BDIR/CMakeCache.txt" "$BDIR/CMakeFiles"
  fi
fi

if [[ $CLEAN -eq 1 ]]; then
  echo "[clean] Removing $BDIR" >&2
  if [[ -e "$BDIR" ]]; then
    # On some filesystems/sync setups, a straight rm -rf can sporadically fail with
    # "Directory not empty" due to concurrent directory entry updates. Use a
    # rename+delete fallback to make --clean robust.
    if ! rm -rf "$BDIR"; then
      echo "[clean] rm -rf failed; attempting rename+delete fallback" >&2
      tmp_dir="${BDIR}.old.$(date +%s).$$"
      if mv "$BDIR" "$tmp_dir" 2>/dev/null; then
        rm -rf "$tmp_dir" || true
      fi
      rm -rf "$BDIR" || true
    fi
    if [[ -e "$BDIR" ]]; then
      echo "[clean] Failed to remove $BDIR" >&2
      exit 1
    fi
  fi
fi

# Try modules before deciding whether CUDA is available. Keep an already
# loaded CUDA toolchain, and never load GPU modules for an explicit CPU build.
detected_toolchain=""
if [[ "$CUDA_MODE" != "off" ]] && ! command -v nvcc >/dev/null 2>&1; then
  # Initialize modules if available
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh || true
  fi
  if ! command -v module >/dev/null 2>&1 && [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash || true
  fi
  if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/lmod.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/lmod.sh || true
  fi
  if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/z00_lmod.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/z00_lmod.sh || true
  fi
  # Fallback: if modulecmd exists, define a minimal module function
  if ! command -v module >/dev/null 2>&1 && command -v modulecmd >/dev/null 2>&1; then
    module() { eval "$(modulecmd bash "$@")"; }
  fi
  if command -v module >/dev/null 2>&1; then
    for module_name in cuda/12.5 accel/cuda/12.5 cuda/12.2 accel/cuda/12.2 accel/nvhpc/24.9; do
      echo "[env] Trying: module load ${module_name}" >&2
      if module load "${module_name}" >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
        echo "[env] Loaded ${module_name}" >&2
        break
      fi
    done
    if ! command -v nvcc >/dev/null 2>&1; then
      echo "[env] No CUDA-capable toolchain found after module load attempts; continuing without it" >&2
    fi
  fi
fi

# Detect CUDA availability if in auto mode
CUDA_AVAILABLE=0
if [[ "$CUDA_MODE" == "auto" || "$CUDA_MODE" == "on" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_AVAILABLE=1
    echo "[detect] CUDA toolchain detected" >&2
  else
    echo "[detect] CUDA toolchain not found" >&2
  fi
fi

# Determine whether to enable CUDA
ENABLE_CUDA=0
if [[ "$CUDA_MODE" == "on" ]]; then
  ENABLE_CUDA=1
  if [[ $CUDA_AVAILABLE -eq 0 ]]; then
    echo "[error] --cuda=on specified but CUDA toolchain not found" >&2
    exit 1
  fi
elif [[ "$CUDA_MODE" == "off" ]]; then
  ENABLE_CUDA=0
  echo "[config] Building CPU-only version (--cuda=off)" >&2
elif [[ "$CUDA_MODE" == "auto" ]]; then
  ENABLE_CUDA=$CUDA_AVAILABLE
  if [[ $ENABLE_CUDA -eq 1 ]]; then
    echo "[config] Auto-detected CUDA: enabling GPU support" >&2
  else
    echo "[config] CUDA not detected: building CPU-only version" >&2
  fi
fi

# Decide compilers to pass to CMake
cmake_args=( -S "$SCRIPT_DIR" -B "$BDIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" )

# Set CUDA mode
if [[ $ENABLE_CUDA -eq 1 ]]; then
  cmake_args+=( -DDMFE_WITH_CUDA=ON )
  # Keep a configured CUDA compiler rather than switching it on a repeat run.
  cached_cuda_compiler=""
  if [[ -f "$BDIR/CMakeCache.txt" ]]; then
    cached_cuda_compiler=$(sed -n 's/^CMAKE_CUDA_COMPILER:[^=]*=//p' "$BDIR/CMakeCache.txt")
  fi
  if [[ -z "$cached_cuda_compiler" ]]; then
    cmake_args+=( -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" )
  fi
else
  cmake_args+=( -DDMFE_WITH_CUDA=OFF )
fi

# Explicit architecture overrides take priority; keep a valid cached choice on
# repeat builds (the login node GPU need not match the compute node GPUs).
cached_arch=""
if [[ -f "$BDIR/CMakeCache.txt" ]]; then
  cached_arch=$(sed -n 's/^CMAKE_CUDA_ARCHITECTURES:[^=]*=//p' "$BDIR/CMakeCache.txt")
fi
if [[ $ENABLE_CUDA -eq 1 && -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
  cmake_args+=( "-DCMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES" )
elif [[ $ENABLE_CUDA -eq 1 && -z "$cached_arch" ]]; then
  gpu_arch=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
      sed -n '1{s/[[:space:]]//g;p;}') || true
    if [[ "$gpu_cap" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
      gpu_arch="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    else
      # Older drivers may not expose compute_cap. NVIDIA's capability table:
      # https://developer.nvidia.com/cuda/gpus
      gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null |
        sed -n '1p' | tr 'A-Z' 'a-z') || true
      case "$gpu_name" in
        *h100*|*hopper*|*sm_90*|*sm90*) gpu_arch=90 ;;
        *a100*|*sm_80*|*sm80*) gpu_arch=80 ;;
        *a10*|*sm_86*|*sm86*) gpu_arch=86 ;;
        *l40*|*ada*|*sm_89*|*sm89*) gpu_arch=89 ;;
        *v100*|*sm_70*|*sm70*) gpu_arch=70 ;;
      esac
    fi
  fi
  # Do not ask an older toolkit to compile architectures it does not support.
  supported_archs=$(nvcc --list-gpu-code 2>/dev/null |
    sed -n 's/^sm_\([0-9][0-9]*\)$/\1/p' | tr '\n' ' ') || true
  if [[ -n "$gpu_arch" ]]; then
    if [[ -n "$supported_archs" && " $supported_archs " != *" $gpu_arch "* ]]; then
      echo "[error] CUDA compiler does not support detected GPU architecture $gpu_arch" >&2
      exit 1
    fi
    echo "[detect] Setting CUDA arch to $gpu_arch based on GPU" >&2
    cmake_args+=( "-DCMAKE_CUDA_ARCHITECTURES=$gpu_arch" )
  else
    portable_archs=""
    for arch in 80 86 89 90; do
      if [[ -z "$supported_archs" || " $supported_archs " == *" $arch "* ]]; then
        portable_archs="${portable_archs:+$portable_archs;}$arch"
      fi
    done
    if [[ -z "$portable_archs" ]]; then
      for arch in $supported_archs; do
        portable_archs="${portable_archs:+$portable_archs;}$arch"
      done
    fi
    cmake_args+=( "-DCMAKE_CUDA_ARCHITECTURES=$portable_archs" )
  fi
fi

# Let CMake retain cached compilers or initialize from the user's environment.
# Automatic preferences are only for a fresh build without an explicit choice.
if [[ -n "${CUDAHOSTCXX:-}" && $ENABLE_CUDA -eq 1 ]]; then
  cmake_args+=( "-DCMAKE_CUDA_HOST_COMPILER=$CUDAHOSTCXX" )
fi
if [[ -n "${CC:-}${CXX:-}${CUDAHOSTCXX:-}" || -f "$BDIR/CMakeCache.txt" ]]; then
  echo "[toolchain] Using requested or cached compilers" >&2
elif [[ $ENABLE_CUDA -eq 1 ]] && command -v nvc++ >/dev/null 2>&1; then
  # Prefer NVHPC if present (works well as CUDA host and for C/C++)
  echo "[toolchain] Using NVHPC compilers (nvc, nvc++, nvfortran if needed)" >&2
  cmake_args+=(
    -DCMAKE_C_COMPILER=nvc
    -DCMAKE_CXX_COMPILER=nvc++
  )
  if [[ $ENABLE_CUDA -eq 1 ]]; then
    cmake_args+=( -DCMAKE_CUDA_HOST_COMPILER=nvc++ )
  fi
else
  # CPU-only build: prefer a fast toolchain similar to CUDA host (Clang + libomp) when available
  echo "[toolchain] CPU-only build: selecting optimized host toolchain" >&2

  # Prefer Clang 14 if present for consistent OpenMP (libomp) and vectorization behavior
  if command -v clang++-14 >/dev/null 2>&1 && command -v clang-14 >/dev/null 2>&1; then
    echo "[toolchain] Selecting Clang 14 toolchain for CPU-only build" >&2
    cmake_args+=(
      -DCMAKE_C_COMPILER=clang-14
      -DCMAKE_CXX_COMPILER=clang++-14
    )
  # Otherwise, fall back to a recent GCC if available
  elif command -v g++-12 >/dev/null 2>&1 && command -v gcc-12 >/dev/null 2>&1; then
    echo "[toolchain] Selecting GCC 12 toolchain for CPU-only build" >&2
    cmake_args+=(
      -DCMAKE_C_COMPILER=gcc-12
      -DCMAKE_CXX_COMPILER=g++-12
    )
  elif command -v g++-11 >/dev/null 2>&1 && command -v gcc-11 >/dev/null 2>&1; then
    echo "[toolchain] Selecting GCC 11 toolchain for CPU-only build" >&2
    cmake_args+=(
      -DCMAKE_C_COMPILER=gcc-11
      -DCMAKE_CXX_COMPILER=g++-11
    )
  else
    echo "[toolchain] Using system default compilers" >&2
  fi
  
  if [[ $ENABLE_CUDA -eq 1 ]]; then
    # Prefer a host compiler known to work with the installed CUDA version
    cuda_ver=""
    if command -v nvcc >/dev/null 2>&1; then
      cuda_ver=$(nvcc --version 2>/dev/null | sed -n 's/^.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*$/\1.\2/p' | head -n1)
    fi
    echo "[detect] CUDA version: ${cuda_ver:-unknown}" >&2

    # For CUDA 11.x, prefer clang++-14 as nvcc host to avoid GCC12/libstdc++ locale issues
    if [[ "$cuda_ver" == 11.* ]] && command -v clang++-14 >/dev/null 2>&1 && command -v clang-14 >/dev/null 2>&1; then
      echo "[toolchain] Selecting Clang 14 toolchain for CUDA 11.x host compiler" >&2
      cmake_args+=(
        -DCMAKE_C_COMPILER=clang-14
        -DCMAKE_CXX_COMPILER=clang++-14
        -DCMAKE_CUDA_HOST_COMPILER=clang++-14
      )
      detected_toolchain="clang-14"
    fi

    if [[ -z "$detected_toolchain" ]]; then
      pick_list=()
      case "$cuda_ver" in
        11.*)
          # CUDA 11.x supports GCC up to 11
          pick_list=(11 10 9 8)
          ;;
        12.*)
          # CUDA 12.x supports newer GCC; prefer 12 then 11
          pick_list=(12 11 10)
          ;;
        *)
          pick_list=(11 10 12 9)
          ;;
      esac

      for v in "${pick_list[@]}"; do
        if command -v g++-$v >/dev/null 2>&1 && command -v gcc-$v >/dev/null 2>&1; then
          echo "[toolchain] Selecting GCC $v toolchain for CUDA host compiler" >&2
          cmake_args+=(
            -DCMAKE_C_COMPILER=gcc-$v
            -DCMAKE_CXX_COMPILER=g++-$v
            -DCMAKE_CUDA_HOST_COMPILER=g++-$v
          )
          detected_toolchain="gcc-$v"
          break
        fi
      done
    fi

    if [[ -z "$detected_toolchain" ]] && command -v clang++-14 >/dev/null 2>&1 && command -v clang-14 >/dev/null 2>&1; then
      echo "[toolchain] Selecting Clang 14 toolchain for CUDA host compiler (fallback)" >&2
      cmake_args+=(
        -DCMAKE_C_COMPILER=clang-14
        -DCMAKE_CXX_COMPILER=clang++-14
        -DCMAKE_CUDA_HOST_COMPILER=clang++-14
      )
      detected_toolchain="clang-14"
    fi
  fi
fi

echo "[configure] cmake ${cmake_args[*]}" >&2
cmake "${cmake_args[@]}"

# nproc is absent on a standard macOS installation.
CORES=${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}
if [[ ! "$CORES" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] Invalid build parallelism: $CORES" >&2
  exit 1
fi

echo "[build] cmake --build $BDIR -j $CORES" >&2
cmake --build "$BDIR" -j "$CORES"
