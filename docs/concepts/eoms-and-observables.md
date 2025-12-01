# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/convolution.svg" alt="Convolution icon"/> EOMs and Observables

and the spherical constraint fixing µ(t) from C(t,t)=1. The concrete prefactors and any thermal/noise terms follow the arXiv paper's conventions; DYNAMITE implements those definitions directly.

## Notes

- The exact expressions and units match the paper; see source under `include/EOMs/` for the hardcoded operators used at runtime.
- The non-stationary (aging) regime requires both time integrals and thus benefits from the sparse 2D grid and renormalized history.

## Stored fields

- `QKv`, `QRv`: discretized correlation/response on the sparse gridy" src="/DYNAMITE/assets/icons/function.svg" alt="Function icon"/> Equations of motion (current model) and observables

We evolve correlation C(t,t') and response R(t,t') after a quench on the non-equidistant grid. Currently, DYNAMITE has the mixed spherical p-spin equations hardcoded, matching the definitions in Lang–Sachdev–Diehl (arXiv:2504.06849). Generalization to pluggable models is planned.

## Mixed spherical p-spin EOMs (paper definitions)

Let the memory kernels be functionals of C and R with p- and q-body terms and spherical constraint enforced via a Lagrange multiplier µ(t). The equations read schematically

\[
\begin{aligned}
\partial_t C(t,t') &= -\mu(t)\, C(t,t') \\
&\quad+ \int_0^t ds\, \Sigma(t,s)\, C(s,t') \\
&\quad+ \int_0^{t'} ds\, D(t,s)\, R(t',s), \\
\partial_t R(t,t') &= -\mu(t)\, R(t,t') + \delta(t-t') \\
&\quad+ \int_{t'}^t ds\, \Sigma(t,s)\, R(s,t'),
\end{aligned}
\]

with model-specific kernels (for a representative mixed p,q case) of the form

\[
\begin{aligned}
\Sigma(t,s) &= p\,J_p^2\, C(t,s)^{p-1} R(t,s) \\
&\quad+ q\,J_q^2\, C(t,s)^{q-1} R(t,s), \\
D(t,s) &= p\,J_p^2\, C(t,s)^{p-1} C(t,s) \\
&\quad+ q\,J_q^2\, C(t,s)^{q-1} C(t,s),
\end{aligned}
\]

and the spherical constraint fixing µ(t) from C(t,t)=1. The concrete prefactors and any thermal/noise terms follow the arXiv paper’s conventions; DYNAMITE implements those definitions directly.

Notes:
- The exact expressions and units match the paper; see source under `include/EOMs/` for the hardcoded operators used at runtime.
- The non-stationary (aging) regime requires both time integrals and thus benefits from the sparse 2D grid and renormalized history.

## Stored fields

- `QKv`, `QRv`: discretized correlation/response on the sparse grid
- `dQKv`, `dQRv`: time derivatives
- `t1grid`: time grid values used by the integrator
- `rvec`, `drvec`: reduced observables stored along the diagonal

See `include/EOMs/*` and `include/interpolation/*` for algorithmic details.
