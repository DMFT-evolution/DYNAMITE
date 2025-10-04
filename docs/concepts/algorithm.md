# <img class="icon icon-lg icon-primary" src="/DMFE/assets/icons/algorithm.svg" alt="Algorithm icon"/> Algorithm (from Lang–Sachdev–Diehl, arXiv:2504.06849)

This section summarizes the numerical renormalization algorithm implemented in DMFE for solving non-stationary dynamical mean-field equations (DMFT) after a quench.

## Dynamical equations

We evolve correlation C(t,t') and response R(t,t') for t ≥ t'. The equations take the schematic form

$$
\partial_t C(t,t') = \mathcal{F}[C,R](t,t')\,,\qquad
\partial_t R(t,t') = \mathcal{G}[C,R](t,t')\,.
$$

For the spherical mixed p-spin model (representative), the functionals \(\mathcal{F},\mathcal{G}\) contain convolution-like memory integrals in both time arguments with kernels determined by the interaction orders and couplings. The renormalization scheme reorganizes these integrals to achieve sublinear scaling in simulated time.

## Numerical renormalization: core idea

- Represent the 2D time plane on an adaptive sparse grid with nested blocks.
- Interleave time evolution with periodic sparsification to prune redundant history while preserving interpolation accuracy.
- Maintain a compressed representation of C and R enabling fast convolution-like updates via precomputed index/weight maps.

This reduces the asymptotic cost from \(\mathcal{O}(T^3)\) to sublinear in the total simulated time (see paper for precise exponents and regimes), enabling orders-of-magnitude longer runs.

Important: DMFE uses exactly the non-equidistant, nested grid defined in the arXiv paper. The performance gains critically rely on this grid; substituting an equidistant grid typically destroys sublinear scaling. See Interpolation grids for details.

## Discrete scheme and data layout

- Store fields on a sparse set of (t, t') nodes; the diagonal t=t' is tracked separately for observables.
- Precompute interpolation structures (positions, indices, weights) on a base grid of length L; these are used for fast 2D interpolation.
- History is organized in layers corresponding to renormalized time scales; layers can be coarsened as t grows while keeping error below a tolerance.

## Symbols in outputs

- `QKv`, `QRv`: grid samples of C and R
- `dQKv`, `dQRv`: their time derivatives
- `t1grid`: time nodes used by the integrator
- `rvec`, `drvec`: reduced observables along the diagonal

## Integrator and update cycle

Time stepping follows the adaptive RK54 default and automatically switches to SSPRK104 once RK54 approaches its stability limit. After each sparsification, the code may attempt SERK2 (can be disabled via CLI). One step:

1. Propose \(\Delta t\) from local error control (tolerance `-e`, min step `-d`).
2. Interpolate required history slices for convolution terms using precomputed indices and weights.
3. Evaluate \(\mathcal{F},\mathcal{G}\) to obtain \(\partial_t C, \partial_t R\).
4. Advance to t+\(\Delta t\) with the active integrator (RK54 or SSPRK104; SERK2 when trialed after sparsify); update diagonal quantities.
5. If pruning is due, sparsify history with a criterion ensuring interpolation error below target.

## Pseudocode

```
Initialize grids and data (C, R) at t = 0
while t < t_max and steps < m_max:
  dt <- propose_step(tolerance=e, min_step=d)
  H <- interpolate_history(C, R, t, dt, structures=L)
  dC, dR <- EOMs(H, params)
  C_next, R_next <- step_active_integrator(C, R, dC, dR, dt)
  update_diagonal_observables(C_next, R_next)
  if sparsify_due(t):
    C, R <- sparsify(C_next, R_next, criterion)
  else:
    C, R <- C_next, R_next
  save_if_needed()
  t <- t + dt
```

## Error control and accuracy

- Local truncation error is controlled via `-e`; monitor step-size independence of observables.
- Sparsification ensures interpolation error is bounded; validate by short runs with sparsification off or conservative thresholds.
- Convergence in L (512/1024/2048) must be checked for quantities of interest, especially near transitions.

## Complexity and performance

The nested sparse representation amortizes the cost of memory integrals, leading to sublinear growth with simulated time. GPU kernels accelerate interpolation and convolution; CPU fallback preserves portability. Asynchronous I/O decouples storage from integration.

## References

- J. Lang, S. Sachdev, M. Diehl, “Numerical renormalization of glassy dynamics,” arXiv:2504.06849.

## See also

- EOMs and Observables: eoms-and-observables.md
- Time Integration: time-integration.md
