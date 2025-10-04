# <img class="icon icon-lg icon-primary" src="/DMFE/assets/icons/tag.svg" alt="Version tag icon"/> Version compatibility

Outputs embed build/version metadata. Resume compatibility is determined by the version policy and can be overridden explicitly.

## Policy

- Identical: code_version in file equals current code_version → resume allowed silently.
- Compatible: major.minor match (first two numeric parts equal) → resume allowed; warnings may be emitted if git branch/hash differ.
- Warning: file version unknown or partial metadata → resume allowed with a warning.
- Incompatible: major.minor differ → resume aborted unless `--allow-incompatible-versions=true` is set.

## Behavior and sources

- Version info sources: `include/version/version_info.hpp` and `src/version/version_info.cpp` (code_version, git hash/branch, compiler, CUDA/OpenMP flags).
- Checks and reporting: `src/version/version_compat.cpp` (functions: analyzeVersionCompatibility, checkVersionCompatibilityInteractive/basic), used during config parsing and when resuming.
- CLI shortcuts: `-c, --check FILE` prints an analysis for a `params.txt` without running; `--allow-incompatible-versions=true|false` toggles override.

## What is compared when resuming

- Primarily the code version string. Additional metadata (git branch/hash, dirty state) are reported as warnings when major.minor are equal but commits differ.
- Runtime options like `--serk2` and sparsification aggressiveness are stored with outputs and re‑applied; mismatches are surfaced in logs even if allowed.

## Recommendation

- Prefer resuming only with compatible versions. If you must override, record the exact git hash and re‑validate observables (energy, equal‑time C, representative C(t, αt)).
