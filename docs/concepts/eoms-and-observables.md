# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/function.svg" alt="Function icon"/> Equations

This page lists the equations that the released program integrates and the quantities it writes out.

## Model

The mixed spherical $p$-spin model consists of $N\to\infty$ real variables $\sigma_i$ with the spherical constraint $\sum_i\sigma_i^2=N$ and the Hamiltonian

$$
H=\sqrt{\lambda}\,H_p+\sqrt{1-\lambda}\,H_s\,,\qquad
H_p=\sum_{i_1,\dots,i_p} J_{i_1\dots i_p}\,\sigma_{i_1}\cdots\sigma_{i_p}\,,
$$

with Gaussian couplings. The only property of the disorder that enters the dynamics is the covariance function

$$
f(q)=\lambda\,q^{p}+(1-\lambda)\,q^{s}\,,
$$

set by the options `-p`, `-q` (for $s$) and `-l` (for $\lambda$). Note that $f(1)=1$; the pure $p$-spin model is $\lambda=1$.

Before $t=0$ the system is in equilibrium at temperature $T_0$ (option `-T`); $T_0=\infty$ is a random initial condition. At $t=0$ it is coupled to a bath at temperature $T$ (option `-G`) and follows Langevin dynamics. In the limit $N\to\infty$ the dynamics is described exactly by the correlation and response functions

$$
C(t,t')=\frac1N\sum_i\overline{\langle\sigma_i(t)\sigma_i(t')\rangle}\,,\qquad
R(t,t')=\frac1N\sum_i\frac{\partial\overline{\langle\sigma_i(t)\rangle}}{\partial h_i(t')}\,,
$$

where $h_i$ is a field conjugate to $\sigma_i$. By causality $R(t,t')=0$ for $t<t'$.

## Equations of motion

With $\beta_0=1/T_0$, for $t>t'$,

$$
\begin{aligned}
\partial_t C(t,t') &= -\mu(t)\,C(t,t') + \int_0^t \mathrm ds\, f''(C(t,s))\,R(t,s)\,C(s,t')
+ \int_0^{t'}\mathrm ds\, f'(C(t,s))\,R(t',s) + \beta_0\,f'(C(t,0))\,C(t',0)\,,\\
\partial_t R(t,t') &= -\mu(t)\,R(t,t') + \int_{t'}^t \mathrm ds\, f''(C(t,s))\,R(t,s)\,R(s,t')\,,
\end{aligned}
$$

with the boundary conditions $C(t,t)=1$ and $R(t,t^-)=1$. The Lagrange multiplier of the spherical constraint is

$$
\mu(t)=T+\int_0^t\mathrm ds\,\Big[f''(C(t,s))\,C(t,s)+f'(C(t,s))\Big]R(t,s)+\beta_0\,f'(C(t,0))\,C(t,0)\,,
$$

and the energy per spin is

$$
E(t)=-\int_0^t\mathrm ds\,f'(C(t,s))\,R(t,s)-\beta_0\,f(C(t,0))\,.
$$

In the notation of the general form on the [home page](../index.md), the kernels are $K(t,s)=f''(C(t,s))\,R(t,s)$ and $D(t,s)=f'(C(t,s))$, and the forcing is $F_C(t,t')=\beta_0 f'(C(t,0))\,C(t',0)$.

## Derivatives along the rays $t'=\theta t$

The solver stores the fields as functions of $t$ and $\theta=t'/t$, $\mathcal C(t,\theta)=C(t,\theta t)$, and advances them in $t$ at fixed $\theta$ (see [Algorithm](algorithm.md)). This requires

$$
\frac{\mathrm d}{\mathrm dt}\,\mathcal C(t,\theta)=\Big[\partial_t C+\theta\,\partial_{t'}C\Big]_{t'=\theta t}\,,
$$

and the same for $R$, so the derivatives with respect to the second argument are needed as well:

$$
\begin{aligned}
\partial_{t'} C(t,t') &= -\mu(t')\,C(t,t') + 2T\,R(t,t') + \int_0^{t'}\mathrm ds\, f''(C(t',s))\,R(t',s)\,C(t,s)
+ \int_0^{t}\mathrm ds\, f'(C(t',s))\,R(t,s) + \beta_0\,f'(C(t',0))\,C(t,0)\,,\\
\partial_{t'} R(t,t') &= \mu(t')\,R(t,t') - \int_{t'}^{t}\mathrm ds\, R(t,s)\,f''(C(s,t'))\,R(s,t')\,.
\end{aligned}
$$

The term $2T\,R(t,t')$ comes from the thermal noise; its counterpart in $\partial_t C$ is proportional to $R(t',t)$ and vanishes for $t>t'$. On the diagonal $\theta=1$ the total derivative vanishes for both $C$ and $R$, so $C(t,t)=1$ and $R(t,t^-)=1$ hold at all times without being imposed. The equations are implemented in `src/EOMs/time_steps.cpp` (CPU) and `src/EOMs/time_steps.cu` (GPU).

## Useful exact results

For checks of the numerics:

- **Pure $p=2$ model** ($\lambda=1$, quench from $T_0=\infty$ to $T=0$): with $J=\sqrt{f''(1)}=\sqrt2$, $b(\tau)=I_1(2J\tau)/(J\tau)$ and $g(t)=\sqrt{b(2t)}$,
  $C(t,t')=b(t+t')/[g(t)g(t')]$ and $E(t)=-J\,I_0(4Jt)/I_1(4Jt)+1/(2t)$. The regression tests in `tests/` compare with these expressions.
- **Threshold energy** for a quench from $T_0=\infty$ to $T=0$ (pure models, and mixed models with weak ergodicity breaking):
  $E_{\rm th}=\dfrac{f(1)-f'(1)}{\sqrt{f''(1)}}-\dfrac{f(1)\sqrt{f''(1)}}{f'(1)}$, which equals $-2\sqrt{(p-1)/p}$ for the pure model.
- **Marginality:** after a quench to $T=0$ into the aging regime, $\mu(t)\to2\sqrt{f''(1)}$ at late times.

$I_0$ and $I_1$ are modified Bessel functions.

## What is stored

The state at every stored time $t_n$ consists of $L$ values per field on the $\theta$ grid. In the checkpoint and the code the arrays are named

| Name | Content |
|---|---|
| `QKv`, `QRv` | $C(t_n,\theta_i t_n)$ and $R(t_n,\theta_i t_n)$ |
| `dQKv`, `dQRv` | their derivatives along the rays, $\frac{\mathrm d}{\mathrm dt}\mathcal C$ and $\frac{\mathrm d}{\mathrm dt}\mathcal R$, multiplied by the length of the preceding time step |
| `t1grid` | the times $t_n$ |
| `rvec`, `drvec` | $\mu(t_n)$ and its derivative (scaled in the same way) |

The text outputs derived from these are described in [Reading outputs](../tutorials/reading-outputs.md).
