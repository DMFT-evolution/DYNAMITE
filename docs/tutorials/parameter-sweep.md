# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/algorithm.svg" alt="Algorithm icon"/> Tutorial: Parameter Sweep

**Audience:** new users who want to launch multiple runs with systematic parameter changes.

Run a simple sweep over `lambda` while reusing grid data.

Example shell loop:

```bash
for L in 512; do
  for lam in 0.4 0.5 0.6; do
    ./RG-Evo -L "$L" -l "$lam" -m 40000 -D false -s true \
      --out-dir "Results/sweep_L${L}"
  done
done
```

Outputs are organized under your chosen out-dir by parameter tuple.

## See also

- [Tutorial → First run](first-run.md)
- [Tutorial → Reading outputs](reading-outputs.md)
