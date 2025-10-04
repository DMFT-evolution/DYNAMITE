# <img class="icon icon-lg icon-primary" src="/DMFE/assets/icons/cpu.svg" alt="CPU icon"/> Tutorial: CPU-only Run

If CUDA is not available or you want deterministic CPU runs, disable the GPU path.

Build as usual, then run:

```bash
./RG-Evo --gpu false -L 512 -l 0.5 -m 20000
```

## Notes

- CPU fallback is always available.
- Performance is lower than GPU; reduce `-m` or `-t` for quick experiments.
