# <img class="icon icon-lg icon-primary" src="/DMFE/assets/icons/gpu.svg" alt="GPU icon"/> Tutorial: Cluster Portable Build

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
