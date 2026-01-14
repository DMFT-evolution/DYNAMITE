# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/function.svg" alt="Function icon"/> Equations of motion (current model) and observables

**Audience:** readers who want the model equations and what is measured/exported.

We evolve correlation $C(t,t')$ and response $R(t,t')$ after a quench on the non-equidistant grid $t\ge t'$. Currently, DYNAMITE has the mixed spherical $p$-spin equations hardcoded, matching the definitions in Lang–Sachdev–Diehl (Phys. Rev. Lett. **135**, 247101 (2025), [doi:10.1103/z64g-nqs6](https://journals.aps.org/prl/abstract/10.1103/z64g-nqs6)).

## Mixed spherical p-spin EOMs (paper definitions)

Let the memory kernels be functionals of C and R with p- and q-body terms and spherical constraint enforced via a Lagrange multiplier µ(t). The equations read schematically

$$
\begin{aligned}
\partial_t C(t,t') &= -\mu(t)\, C(t,t') \\
&\quad+ \int_0^t ds\, \Sigma(t,s)\, C(s,t') \\
&\quad+ \int_0^{t'} ds\, D(t,s)\, R(t',s), \\
\partial_t R(t,t') &= -\mu(t)\, R(t,t') + \delta(t-t') \\
&\quad+ \int_{t'}^t ds\, \Sigma(t,s)\, R(s,t'),
\end{aligned}
$$

with model-specific kernels (for the mixed spherical $p$-spin model) of the form

$$
\begin{aligned}
\Sigma(t,s) &= f''(C(t,s))R(t,s),\\
D(t,s) &= f'(C(t,s)) + \beta\, \delta(s)\, f'(C(t,0))\,,
\end{aligned}
$$

where

$$
\begin{aligned}
f(x)=J_p^2\, x^p + J_q^2\, x^q\,,
\end{aligned}
$$

and the Lagrange multiplier of the spherical constraint

$$
\begin{aligned}
\mu(t)=T_f+\int_0^t ds\left[\Sigma(t,s)C(t,s)+D(t,s)R(t,s)\right]\,.
\end{aligned}
$$

## Notes
- The spherical constraint fixes $\mu(t)$ such that $C(t,t)=1$.
- The concrete prefactors and any thermal/noise terms follow the conventions published in Phys. Rev. Lett.; DYNAMITE implements those definitions directly.
- The exact expressions and units match the paper; see source under [`include/EOMs/`](https://github.com/DMFT-evolution/DMFE/tree/main/include/EOMs) for the hardcoded operators used at runtime.
- The non-stationary (aging) regime requires both time integrals and thus benefits from the sparse 2D grid and renormalized history.

## Stored fields

- `QKv`, `QRv`: discretized correlation/response on the sparse grid
- `dQKv`, `dQRv`: time derivatives
- `t1grid`: time grid values used by the integrator
- `rvec`, `drvec`: reduced observables stored along the diagonal

See [`include/EOMs/`](https://github.com/DMFT-evolution/DMFE/tree/main/include/EOMs) and [`include/interpolation/`](https://github.com/DMFT-evolution/DMFE/tree/main/include/interpolation) for algorithmic details.
