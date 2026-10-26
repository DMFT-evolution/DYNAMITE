# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/sparsify.svg" alt="Sparsification icon"/> Sparsification

Every accepted time step adds a slice $C(t_n,\theta_i t_n)$, $R(t_n,\theta_i t_n)$, $i=1,\dots,L$, to the stored history. Kept in full, the history grows linearly with the number of steps, which at late times exceeds the memory of a GPU. Many of these slices are not needed: the history is only used through the cubic Hermite interpolation in $t$ (see [Grid and interpolation](interpolation-grids.md)), and where the fields vary slowly, a slice can be reconstructed from its neighbours. DYNAMITE therefore removes such slices periodically. The code is in `src/sparsify/`.

## Criterion

Consider slice $n$ and its neighbours $n-1$ and $n+1$, with $H=t_{n+1}-t_{n-1}$. For each field $\mathcal A\in\{\mathcal C,\mathcal R\}$ and each ray $\theta_i$ the code evaluates

$$
\Delta^{\mathcal A}_n(\theta_i)=\frac{H}{12}\Big[2\big(\mathcal A_{n+1,i}-\mathcal A_{n-1,i}\big)-H\big(\dot{\mathcal A}_{n-1,i}+\dot{\mathcal A}_{n+1,i}\big)\Big]\,,
$$

where $\dot{\mathcal A}$ is the derivative along the ray. Only the two neighbours enter: $\Delta^{\mathcal A}_n$ compares the change of $\mathcal A$ across the interval with the trapezoidal estimate $\tfrac H2(\dot{\mathcal A}_{n-1}+\dot{\mathcal A}_{n+1})$, and is of order $H^4\,\partial_t^3\mathcal A$. It is small when the cubic Hermite interpolant through the two neighbours describes the interval well, so that slice $n$ adds little information. Slice $n$ is removed if

$$
\epsilon_{\rm sparse}(n)=\sum_{i=1}^{L}\Big(\big|\Delta^{\mathcal C}_n(\theta_i)\big|+\big|\Delta^{\mathcal R}_n(\theta_i)\big|\Big)<\epsilon\,,
$$

with the same tolerance $\epsilon$ as for the Runge–Kutta steps (option `-e`).

## Procedure

- Within one sweep, no two neighbouring slices are removed, so each removal is judged against neighbours that are kept. On the GPU only every second slice is considered; on the CPU every slice is considered, but after a removal the next slice is kept.
- The first slice and the last two slices are always kept.
- The derivatives are stored multiplied by the length of the preceding time interval (see [Equations](eoms-and-observables.md)). After a removal they are rescaled to the new, longer interval, so that the reduced history is again a consistent cubic Hermite representation.
- Sparsification is part of the maintenance pass, which runs every `--sparsify-interval` steps (default 100000). Each pass performs up to `-w` sweeps and stops early if a sweep removes nothing. Afterwards the interpolation is recomputed and a checkpoint is written.

## Settings

| `-w` | Effect |
|---|---|
| `0` | No sparsification. Memory grows linearly with the number of steps. |
| `1`, `2`, … | That many sweeps per maintenance pass (at most 10). |
| `-1` (default) | One sweep, or two on a GPU whose memory is more than half full. |

Because the default depends on the memory used on the GPU, also by other processes, set `-w` explicitly for runs that must be reproducible.

## Effect on accuracy

Each removed slice changes the history by less than $\epsilon$ in the sense above, the same bound as for a single Runge–Kutta step. The criterion does not control how such a change propagates in the subsequent evolution; that is monitored by the Runge–Kutta error estimate. A direct check is to repeat a run, or its first part, with `-w 0` and compare the observables of interest. In the benchmarks of the [DYNAMITE paper](https://arxiv.org/abs/2604.06309), sparsification reduces the growth of the memory to roughly $t^{1/3}$.
