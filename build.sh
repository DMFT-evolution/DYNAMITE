#!/usr/bin/env bash
set -euo pipefail

# Usage: ./build.sh [build-dir] [--clean]
# Default build dir: build

BDIR=${1:-build}
CLEAN=0
if [[ "${*: -1}" == "--clean" ]]; then
  CLEAN=1
fi

if [[ $CLEAN -eq 1 ]]; then
  echo "[clean] Removing $BDIR" >&2
  rm -rf "$BDIR"
fi

if [[ ! -d $BDIR ]]; then
  echo "[configure] cmake -S . -B $BDIR" >&2
  cmake -S . -B "$BDIR"
fi

CORES=$(nproc 2>/dev/null || echo 4)

echo "[build] cmake --build $BDIR -j $CORES" >&2
cmake --build "$BDIR" -j "$CORES"
