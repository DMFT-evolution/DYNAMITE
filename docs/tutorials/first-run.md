# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/file.svg" alt="Tutorial icon"/> Tutorial: First run

This tutorial builds the program, runs a short simulation on the CPU and compares it with the exact solution. No GPU is needed.

## 1. Build

```bash
git clone https://github.com/DMFT-evolution/DYNAMITE.git
cd DYNAMITE
./build.sh
```

This takes a few minutes and produces the executable `RG-Evo`. If the build fails, see [Install](../install.md); on a machine without CUDA, `./build.sh --cuda=off` is enough.

## 2. Run

We simulate the pure 2-spin model ($\lambda=1$), quenched from a random initial condition ($T_0=\infty$) to zero temperature, up to $t=1$. For this case the exact solution is known.

```bash
OMP_NUM_THREADS=8 ./RG-Evo -p 2 -q 3 -l 1 -T inf -G 0 -e 1e-8 -t 1 -m 1e6 -g false -D false -o Results/
```

| Option | Meaning |
|---|---|
| `-p 2 -q 3 -l 1` | $f(q)=q^2$: with $\lambda=1$ the second order `-q` plays no role |
| `-T inf -G 0` | random initial condition, bath at temperature 0 |
| `-e 1e-8` | tolerance of the time stepping; the default is the stricter `1e-10` |
| `-t 1 -m 1e6` | stop at $t=1$; allow up to $10^6$ steps |
| `-g false` | run on the CPU, also if a GPU is present |
| `-D false` | a single status line instead of one line per step |
| `-o Results/` | write the results below `Results/` |

On a desktop computer (Intel Core i7-13700, 8 threads) the run takes less than a minute; on a GPU a few seconds. The program prints where it writes its output, e.g.

```
Results/26_10_2026_1534_p=2_q=3_lambda=1.00_T0=inf_G=0.00_L=512/
```

It uses the grid in `Grid_data/512/`, which is part of the repository.

## 3. Look at the result

`correlation.txt` in the run directory has three columns: the time $t$, the energy $E(t)$ and $C(t,0)$. For this model

$$
C(t,0)=\frac{b(t)}{\sqrt{b(2t)}}\,,\qquad b(\tau)=\frac{I_1(2\sqrt2\,\tau)}{\sqrt2\,\tau}\,,
$$

with the modified Bessel function $I_1$. The script below plots the numerical result and its deviation from the exact one (it needs `numpy`, `scipy` and `matplotlib`):

```python
import numpy as np, matplotlib.pyplot as plt
from scipy.special import ive

t, E, C0 = np.loadtxt("Results/<run>/correlation.txt", unpack=True)
later_min = np.minimum.accumulate(t[::-1])[::-1]          # drop steps discarded by a rollback
keep = np.append(t[:-1] < later_min[1:], True)
t, C0 = t[keep], C0[keep]

J = np.sqrt(2)
def b(tau):                       # I_1(2 J tau)/(J tau), written with the scaled Bessel function
    return ive(1, 2*J*tau) * np.exp(2*J*tau) / (J*tau)
exact = b(t) / np.sqrt(b(2*t))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(t, C0); ax[0].set_xlabel("t"); ax[0].set_ylabel("C(t,0)")
ax[1].semilogy(t, abs(C0 - exact)); ax[1].set_xlabel("t"); ax[1].set_ylabel("error")
plt.tight_layout(); plt.show()
```

The error stays below about $10^{-12}$, far below the tolerance: `-e` bounds the error of a single step, and in this short run the step size is limited by how fast it may grow (1% every second step), not by the tolerance.

## 4. Next steps

- Run the mixed model, e.g. `-p 3 -q 4 -l 0.5`, and look at the approach of the energy to its threshold value with `scripts/plot_energy_threshold_gap.py` ([Reading outputs](reading-outputs.md)).
- Running the same command again with a larger `-t` continues the run instead of starting from scratch ([Usage](../usage.md#continuing-a-run)).
- How long runs behave on a CPU, and how to use the cores of your machine, is described in [Running without a GPU](cpu-only.md).
- The options are listed in [Usage](../usage.md), the method in [Algorithm](../concepts/algorithm.md).
