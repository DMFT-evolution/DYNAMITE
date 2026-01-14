# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/convolution.svg" alt="Kernel icon"/> How-to: Add a New Kernel

**Audience:** developers adding performance-critical CPU/CUDA kernels.

This repo has paired CPU/CUDA implementations for performance-critical kernels (convolution, interpolation/search helpers, etc.).
This page shows the “house style” for adding a new one.

## 1) Decide the interface

Keep the call signature identical across CPU and GPU builds.

Typical pattern:

- Public declaration in `include/<module>/<name>.hpp`.
- Implementation in `src/<module>/<name>.cpp`.
- Optional CUDA implementation in `src/<module>/<name>.cu`.

If the kernel is used from both host and device code, add a small wrapper API and keep device details internal.

## 2) Add the CPU implementation

1. Create the header under `include/`.
2. Implement in `src/`.
3. Prefer existing utilities:
	- math helpers: `include/math/*`
	- memory/telemetry: `include/core/gpu_memory_utils.hpp`, `include/core/console.hpp`
	- convolution scaffolding: `include/convolution/*` and `src/convolution/*`

## 3) Add the CUDA implementation (optional)

1. Create `src/<module>/<name>.cu`.
2. Guard CUDA-only bits with `#if DMFE_WITH_CUDA`.
3. Reuse device utilities from `include/core/*` (and `include/core/device_utils.cuh` where appropriate).

Practical advice:

- Keep kernel launches and device memory management localized.
- If the kernel needs to run in both modes depending on `--gpu`, keep the decision at a higher level and provide both paths.

## 4) Wire it into the build

Register the new translation unit(s) in `CMakeLists.txt`:

- add `src/<module>/<name>.cpp` to the CPU sources list
- if applicable, add `src/<module>/<name>.cu` to the GPU sources list

Use existing kernel files as templates (the `src/convolution/` and `src/interpolation/` folders are good starting points).

## 5) Validate correctness and performance

Minimum checks:

- CPU vs GPU: compare results on a small grid with `--gpu false` vs `--gpu true`.
- Stability: ensure no NaNs/inf are produced.
- Performance: time at least one representative call site (or check `step_metrics.txt` in debug mode).

If it’s a user-visible enhancement, add a short note to the relevant concept/tutorial page.

## See also

- [Concepts → Architecture](../concepts/architecture.md)
- [Developer → Testing](../dev/testing.md)
