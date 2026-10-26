# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/time.svg" alt="Time integration icon"/> Time integration

The fields $C(t,\theta_i t)$ and $R(t,\theta_i t)$ on the $L$ rays are advanced in $t$ with explicit Runge–Kutta methods and an adaptive step size. The code is in `src/EOMs/runge_kutta.{cpp,cu}` and `src/simulation/simulation_runner.cpp`.

## Methods

- **Dormand–Prince** (RK5(4), 7 stages): used at early times. Its order is high, but its stability interval on the negative real axis only reaches about $-3.3$.
- **SSPRK(10,4)**: a fourth-order strong-stability-preserving method with 10 stages and a stability interval about four times longer. Used once the step size is limited by stability. Strong stability preservation keeps unphysical oscillations from growing, which is what limits the step size at late times.
- **SERK2** (GPU only, option `-S`, on by default): a ladder of second-order stabilized methods with larger stability intervals, which the GPU backend climbs step by step at late times. These methods are not strong-stability-preserving; if a run develops noise at late times, try `-S false`.

All methods are explicit. Implicit methods would allow larger steps, but at late times the equations become increasingly ill-conditioned, and the small tolerances needed limit the step size anyway.

## Stability limit

DYNAMITE estimates the spectral radius of the Jacobian of the discretized equations as

$$
\varrho\simeq 4\sqrt{f''(1)}
$$

and keeps the step size below $r_m/\varrho$, where $r_m$ is the length of the stability interval used for method $m$ (3 for Dormand–Prince, 13 for SSPRK(10,4)). Where this estimate is too optimistic, the error control described below limits the step size.

## Error estimate

Each method provides two solutions of different order. Their difference on the grid, $\delta C(t,\theta_i t)$ and $\delta R(t,\theta_i t)$, gives the error estimate

$$
\epsilon_{\rm RK}=\sum_{i=1}^{L}\Big(\big|\delta C(t,\theta_i t)\big|+\big|\delta R(t,\theta_i t)\big|\Big)\,,
$$

which is compared with the tolerance $\epsilon$ (option `-e`, default $10^{-10}$). This sum is the default (`--error-norm sum`). It grows with $L$, so a fixed $\epsilon$ corresponds to a pointwise tolerance of order $\epsilon/(2L)$. With `--error-norm mean`, `rms` or `max` the sum is replaced by the mean, the root mean square or the maximum of the $2L$ components.

## Step-size control

The evolution of aging systems is slow and has no sudden events, so a simple controller suffices. After every step:

- If $\epsilon_{\rm RK}<\epsilon$, the step size is increased by a factor 1.01, provided that the error is not growing ($\epsilon_{\rm RK}<1.1\,\epsilon_{\rm RK}^{\rm prev}$, or $\epsilon_{\rm RK}^{\rm prev}<\epsilon/1000$), the new step stays below the stability limit $r_m/\varrho$, and the previous step was not itself changed. The step size therefore grows by at most 1% every second step.
- If $\epsilon_{\rm RK}>2\epsilon$, the step is rejected: the last ten time slices are discarded, the step size is halved (and limited to half the stability limit), and the integration continues with SSPRK(10,4); on the GPU, a run that is on the SERK2 ladder moves one rung down instead. If the following step is rejected as well, the step size is halved again.
- Otherwise the step size is kept.

A step with $\epsilon\le\epsilon_{\rm RK}\le2\epsilon$ is thus accepted. The tolerance is a target rather than a strict bound per step; the error of an accepted step never exceeds $2\epsilon$.

The integrator is changed to a more stable one only during a maintenance pass (every `--sparsify-interval` steps, by default 100000) and only if $\epsilon_{\rm RK}<\epsilon/2$ and no step was rejected during the last 1000 steps. The step size is then halved, and Dormand–Prince is replaced by SSPRK(10,4), or, on the GPU with `-S true`, SSPRK(10,4) by the next SERK2 method. A run starts with Dormand–Prince, or with SSPRK(10,4) if the initial step `-d` already exceeds the Dormand–Prince stability limit.

The CPU and the GPU backends use the same controller. The CPU backend has no SERK2 methods, so there the last step of the ladder is SSPRK(10,4).

## The run stops

- at the final time `-t`,
- after `-m` steps,
- or if the step size falls below `-d`. This happens when errors from the interpolation have grown large enough to force ever smaller steps. It shows up as high-frequency noise of $C$ and $R$ on the $\theta$ grid. Lowering `-e`, increasing $L$ or lowering the interpolation order delays this.

## Practical advice

- Check that the observables you need do not change when `-e` is lowered by an order of magnitude.
- Errors made early are carried along by the memory integrals, and at late times the number of steps grows linearly with $t$. Runs to late times therefore need smaller tolerances than short runs.
