# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/tag.svg" alt="Version tag icon"/> Version compatibility

**Audience:** users who need to interpret resume warnings/errors and version policy.

Outputs embed build/version metadata. Resume compatibility is determined by the version policy and can be overridden explicitly.

## Policy

- Identical: code_version in file equals current code_version → resume allowed silently.
- Compatible: major.minor match (first two numeric parts equal) → resume allowed; warnings may be emitted if git branch/hash differ.
- Warning: file version unknown or partial metadata → resume allowed with a warning.
- Incompatible: major.minor differ → resume aborted unless an explicit override is enabled.

## Behavior and sources

- Version info sources: `include/version/version_info.hpp` and `src/version/version_info.cpp` (code_version, git hash/branch, compiler, CUDA/OpenMP flags).
- Checks and reporting: `src/version/version_compat.cpp` (functions: analyzeVersionCompatibility, checkVersionCompatibilityInteractive/basic), used during config parsing and when resuming.
- Practical configuration (including the resume override and any helper shortcuts) is documented in [Usage](../usage.md).

## What is compared when resuming

- Primarily the code version string. Additional metadata (git branch/hash, dirty state) are reported as warnings when major.minor are equal but commits differ.
- Runtime options (integrator choices, interpolation mode, sparsification aggressiveness) are stored with outputs and re‑applied; mismatches are surfaced in logs even if resuming is allowed.

## Recommendation

- Prefer resuming only with compatible versions. If you must override, record the exact git hash and re‑validate observables (energy, equal‑time C, representative C(t, αt)).
