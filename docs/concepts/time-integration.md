# <img class="icon icon-lg icon-primary" src="/DMFE/assets/icons/time.svg" alt="Time integration icon"/> Time integration

Default strategy: adaptive Dormand–Prince RK54. When the adaptive step approaches the RK54 stability limit, the solver switches to SSPRK(10,4) for stability at larger steps. After each sparsification event, the code may attempt SERK2; this trial can be disabled via CLI.

## Summary

- RK54 = Dormand–Prince (adaptive default): embedded error estimate controls local error and proposes Δt.
- SSPRK(10,4) (auto‑switch): engaged when RK54 reaches its absolute‑stability bound to continue with stable steps at late times.
- SERK2 (optional): attempted after sparsification to exploit extended stability; can be turned off with the command‑line flag `--serk2=false`.

## Error and stability controls (paper supplemental and code)

- Error bound on each step applies to the 1‑norm of C and R:

$$
\|C(t,\, \phi_k\, t)\|_1 + \|R(t,\, \phi_k\, t)\|_1 \leq \varepsilon.
$$

- Dormand–Prince stability along the negative real axis extends to about $-3.307$. With an upper bound on the Jacobian spectral radius $\lambda_{\max} \lesssim 4\,\Sigma''(1)$, the code switches to SSPRK$(10,4)$ once $\lambda_{\max}\,\Delta t \gtrsim 3.0$. On switch, $\Delta t$ is halved once to account for the lower order, then adapted normally.
- Step‑size controller increases $\Delta t$ slightly when below target error and reduces it when above (simple proportional strategy; details in supplemental text).

## CLI controls

- `-e` error tolerance (ε) for the embedded estimate.
- `-d` minimum time step to avoid over‑refinement near initial transients.
- `-m`, `-t` cap total iterations and physical time.
- `--serk2=true|false` enables/disables SERK2 trials after sparsification.

## Practice

- Check step‑size independence of key observables (C, R) at representative times.
- If SERK2 trials are disabled, expect exclusively Dormand–Prince and SSPRK$(10,4)$.

Implementation details: see `include/EOMs/` and `src/EOMs/` (RK54 Dormand–Prince, SSPRK(10,4), SERK2 kernels and selection logic).
