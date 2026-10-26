# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/algorithm.svg" alt="Algorithm icon"/> Tutorial: Series of runs

Every run writes into its own directory, named after its parameters, so a series of runs needs nothing more than a loop. For example, quenches of the mixed 3+4 model from different initial temperatures $T_0$ to $T=0$:

```bash
for T0 in 1.0 1.05 1.1 inf; do
  ./RG-Evo -p 3 -q 4 -l 0.5 -T "$T0" -G 0 -t 1000 -m 1e6 -D false -o Results/sweep/
done
```

This produces directories such as

```
Results/sweep/26_10_2026_1612_p=3_q=4_lambda=0.50_T0=1.00000_G=0.00_L=512/
Results/sweep/26_10_2026_1630_p=3_q=4_lambda=0.50_T0=1.05000_G=0.00_L=512/
...
```

The parameters of each run are also listed in its `params.txt`, which is the safest way to identify a run in an analysis script.

Some points to keep in mind:

- Running the loop a second time with a larger `-t` continues each run instead of starting new ones (see [Continuing a run](../usage.md#continuing-a-run)).
- On a single GPU, run the jobs one after the other. On a CPU, several runs can be started at the same time, each with a share of the cores, e.g. `OMP_NUM_THREADS=4 ./RG-Evo ... &`.
- For convergence checks, add runs with a larger `-L` and a smaller `-e` at a few points of the series rather than for all of them.
- For quick plots across a series, `correlation.txt` of each run contains $t$, $E(t)$ and $C(t,0)$; see [Reading outputs](reading-outputs.md).
