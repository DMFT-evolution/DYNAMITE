# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/gpu.svg" alt="GPU icon"/> Tutorial: Cluster Portable Build

**Audience:** users who need a portable build for heterogeneous cluster nodes.

Build and run on heterogeneous cluster nodes without `-march=native` and with shared cudart.

Configure:

```bash
cmake -S . -B build -DDMFE_PORTABLE_BUILD=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j $(nproc)
```

Run:

```bash
./RG-Evo -L 1024 -l 0.5
```

## Tips

- Pick the lowest common SM (e.g., 80 for A100).
- For CPU-only environments, set `-DDMFE_WITH_CUDA=OFF`.

## See also

- [Tutorial → First run](first-run.md)
- [Tutorial → CPU-only run](cpu-only.md)
