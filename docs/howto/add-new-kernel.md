# How-to: Add a New Kernel

1) Start with a header in `include/` and CPU path `.cpp` in `src/`.
2) Add a CUDA path `.cu` in `src/` guarded by `DMFE_WITH_CUDA`.
3) Register files in `CMakeLists.txt` under CPU_SOURCES and GPU_SOURCES.
4) Ensure host/device utilities (`include/core/*`) are used consistently.
5) Benchmark on a small grid; add a tutorial snippet if useful.
