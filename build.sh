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

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RAW_DIR=""
ROOT_BUILD_DIR="build"

# Parse arguments: optional build dir + flags
CLEAN=0
CUDA_MODE="auto"  # auto, on, or off
# Optional: honor CMAKE_BUILD_TYPE if exported; default to Release
BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
for arg in "$@"; do
  case "$arg" in
    --clean)
      CLEAN=1 ;;
    --cuda=auto|--cuda=on|--cuda=off)
      CUDA_MODE="${arg#--cuda=}" ;;
    --)
      shift; break ;;
    -*) ;;
    *)
      if [[ -z "$RAW_DIR" ]]; then RAW_DIR="$arg"; fi ;;
  esac
done

if [[ -z "${RAW_DIR}" || "${RAW_DIR}" == "build" || "${RAW_DIR}" == "--clean" ]]; then
  BDIR="${ROOT_BUILD_DIR}"
elif [[ "${RAW_DIR}" == /* ]]; then
  BDIR="${RAW_DIR}"
elif [[ "${RAW_DIR}" == build/* ]]; then
  BDIR="${RAW_DIR}"
else
  BDIR="${ROOT_BUILD_DIR}/${RAW_DIR}"
fi

if [[ $CLEAN -eq 1 ]]; then
  echo "[clean] Removing $BDIR" >&2
  rm -rf "$BDIR"
fi

# Detect CUDA availability if in auto mode
CUDA_AVAILABLE=0
if [[ "$CUDA_MODE" == "auto" || "$CUDA_MODE" == "on" ]]; then
  if command -v nvcc >/dev/null 2>&1 || command -v nvc++ >/dev/null 2>&1; then
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

# Try to load NVHPC module if CUDA is enabled and nvc++ is missing
detected_toolchain=""
if [[ $ENABLE_CUDA -eq 1 ]] && ! command -v nvc++ >/dev/null 2>&1; then
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
    echo "[env] Trying: module load accel/nvhpc/24.9" >&2
    if module load accel/nvhpc/24.9 >/dev/null 2>&1; then
      echo "[env] Loaded accel/nvhpc/24.9" >&2
    else
      echo "[env] Module accel/nvhpc/24.9 not available or failed to load; continuing without it" >&2
    fi
  fi
fi

# Decide compilers to pass to CMake
cmake_args=( -S "$SCRIPT_DIR" -B "$BDIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" )

# Set CUDA mode
if [[ $ENABLE_CUDA -eq 1 ]]; then
  cmake_args+=( -DDMFE_WITH_CUDA=ON )
  # Explicitly set CUDA compiler if nvcc is available
  if command -v nvcc >/dev/null 2>&1; then
    cmake_args+=( -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" )
  fi
else
  cmake_args+=( -DDMFE_WITH_CUDA=OFF )
fi

# If not explicitly set and CUDA is enabled, try to pick a reasonable CUDA arch for the current GPU
if [[ $ENABLE_CUDA -eq 1 ]] && [[ -z "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
  gpu_arch=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | tr 'A-Z' 'a-z') || true
    case "$gpu_name" in
      *h100*|*hopper*|*sm_90*|*sm90*) gpu_arch=90 ;;
      *a100*|*sm_80*|*sm80*) gpu_arch=80 ;;
      *a10*|*l40*|*ada*|*sm_89*|*sm89*) gpu_arch=89 ;;
      *v100*|*sm_70*|*sm70*) gpu_arch=70 ;;
    esac
  fi
  if [[ -n "$gpu_arch" ]]; then
    echo "[detect] Setting CUDA arch to $gpu_arch based on GPU" >&2
    cmake_args+=( -DCMAKE_CUDA_ARCHITECTURES=$gpu_arch )
  else
    # Fallback portable set if unknown
    cmake_args+=( -DCMAKE_CUDA_ARCHITECTURES=80\;86\;89\;90 )
  fi
fi

# Compiler selection (only matters when CUDA is enabled or for C/C++ in CPU-only mode)
if [[ $ENABLE_CUDA -eq 1 ]] && command -v nvc++ >/dev/null 2>&1; then
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

CORES=$(nproc 2>/dev/null || echo 4)

echo "[build] cmake --build $BDIR -j $CORES" >&2
cmake --build "$BDIR" -j "$CORES"
