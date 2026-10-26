# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/function.svg" alt="Function icon"/> Adapting the code to another model

The method applies to any causal two-time equations of the form shown on the [home page](../index.md): a local term $-\mu(t)A(t,t')$, memory integrals with kernels that depend on $C$ and $R$, and possibly a forcing term. The released program, however, has the equations of the mixed spherical $p$-spin model built in; there is no interface for supplying other kernels. Changing the model means editing the source files listed below. The grid generation, the interpolation, the quadrature, the search routines, the sparsification and the time integration contain nothing specific to the model and need no changes.

## Where the model enters

1. **The covariance function** $f$ and its first three derivatives: `flambda`, `Dflambda`, `DDflambda`, `DDDflambda` in `src/math/math_ops.cpp`, and their GPU counterparts in `src/math/math_ops.cu`. The model parameters $p$, $s$, $\lambda$ are copied to GPU constant memory by `copyParametersToDevice`.
2. **The memory kernels** $D=f'(C)$ (`SigmaK`) and $K=f''(C)\,R$ (`SigmaR`), and their derivatives with respect to $C$ and $R$ (`SigmaK10`, `SigmaK01`, `SigmaR10`, `SigmaR01`), in `src/math/math_sigma.cpp` and `src/math/math_sigma.cu`. The GPU file contains a second, inlined copy of the kernels in `eval_sigmaK_sigmaR_fused`.
3. **The equations of motion**: which memory integrals enter the right-hand sides for $C$ and $R$, the local terms, the noise term $2T\,R$ and the forcing by the initial condition, in `QKstep`, `QRstep` (CPU) and `QKRstepGPU` (GPU) in `src/EOMs/time_steps.{cpp,cu}`. The same files compute $\mu(t)$ (`rstep`) and its time derivative (`drstep`, `drstep2`), i.e. the spherical constraint.
4. **The energy**, $E(t)=-\int f'(C)R-\beta_0 f(C(t,0))$, which is evaluated for the output in `src/EOMs/time_steps.{cpp,cu}` and again in several places in `src/io/`.
5. **The stability estimate** $\varrho=4\sqrt{f''(1)}$ in `src/core/initialization.cpp`, which limits the step size (see [Time integration](time-integration.md)).

The equations along the rays require the derivatives with respect to both time arguments (see [Equations](eoms-and-observables.md)). For a new model these must be worked out first.

## Checking a modified model

- Compare the CPU and the GPU backend with each other on a short run; they should agree to within the tolerance.
- If the model has an exactly solvable limit, compare with it, as the regression test `tests/run_regression.py` does for the pure $p=2$ model.
- Check that $C(t,t)=1$ and $R(t,t^-)=1$ remain satisfied, and that the results converge when $L$ is increased and `-e` is decreased.

