# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/algorithm.svg" alt="Algorithm icon"/> Algorithm

This page summarizes the method implemented in DYNAMITE. A detailed account, with benchmarks, is given in the [DYNAMITE paper](https://arxiv.org/abs/2604.06309); the method was first introduced in J. Lang, S. Sachdev, S. Diehl, Phys. Rev. Lett. **135**, 247101 (2025) [[arXiv:2504.06849](https://arxiv.org/abs/2504.06849)]. The following pages give the details: [Grid and interpolation](interpolation-grids.md), [Time integration](time-integration.md) and [Sparsification](sparsification.md).

## The problem

The equations of motion ([Equations](eoms-and-observables.md)) contain integrals over the whole past. On an equidistant time grid with $N$ steps, each step requires $\mathcal O(N^2)$ operations to evaluate these integrals for all $t'$, so the total cost grows as $N^3$, and storing $C$ and $R$ requires $\mathcal O(N^2)$ memory. In glassy systems this cannot be avoided by truncating the memory: because of aging the system never becomes stationary, and the dynamics at late times still depends on the whole history.

## Discretization

Even at times $t$ much larger than any microscopic scale $\tau_{\rm micro}$, the two-time functions vary quickly in the relative time $\tau=t-t'$ for $\tau\lesssim\tau_{\rm micro}$, and after a quench also in $t'$ for $t'\lesssim\tau_{\rm micro}$. In between they vary on scales that grow with $t$. DYNAMITE therefore writes

$$
A(t,t')=\mathcal A(t,\theta)\,,\qquad \theta=t'/t\in[0,1]\,,
$$

and samples $\theta$ on a fixed grid $\theta_1=0<\theta_2<\dots<\theta_L=1$ that is dense near both ends. The precise choice of the $\theta_i$ is not important as long as the microscopic scales stay resolved,

$$
\theta_{i+1}-\theta_i\ll\frac{\tau_{\rm micro}}{t}\quad\text{wherever}\quad\min(\theta_i,1-\theta_i)\lesssim\frac{\tau_{\rm micro}}{\min(t',\tau)}\,,
$$

for all simulated times $t$. The times $t_n$ themselves are chosen by an adaptive Runge–Kutta method, with small steps during transients and large steps when the evolution is slow. The stored values thus lie on the points $(t_n,\theta_i t_n)$: dense near the diagonal and near $t'=0$, and sparser in between.

Keeping $L$ fixed while $t$ grows means that the resolution in $t'$ becomes coarser in absolute terms as the system ages, while the resolution of the microscopic scales near the two edges is maintained. In the language of the renormalization group this is a coarse-graining step, which is why the method is called a numerical renormalization.

The grid could in principle be made adaptive as well, but this has not been necessary in any of our simulations: only the discretization in $t$ is adaptive.

## Memory integrals

With $t'=\theta_i t$, the two types of memory integrals in the equations of motion become

$$
\begin{aligned}
\int_0^{t'}\mathrm ds\,A(t,s)\,B(t',s) &= t\,\theta_i\int_0^1\mathrm du\;\mathcal A\big(t,\theta_i u\big)\;\mathcal B\big(\theta_i t,\,u\big)\,,\\
\int_{t'}^{t}\mathrm ds\,A(t,s)\,B(s,t') &= t\,(1-\theta_i)\int_0^1\mathrm du\;\mathcal A\big(t,\phi\big)\;\mathcal B\big(\phi\, t,\,\theta_i/\phi\big)\Big|_{\phi=\theta_i+(1-\theta_i)u}\,.
\end{aligned}
$$

The integrals over $u$ are evaluated with the same $\theta$ grid as quadrature nodes, $u=\theta_j$, and quadrature weights $\nu_j$ from a spline (quintic by default). This needs the fields at the points

$$
\phi^{(1)}_{ij}=\theta_i\,\theta_j\,,\qquad \phi^{(2)}_{ij}=\theta_i+(1-\theta_i)\,\theta_j\,,
$$

which are again dense where needed, near $\phi=0$ and $\phi=1$. They are obtained by interpolation:

- **in $\theta$**, with local polynomial interpolation of order 9 by default. The $\phi^{(1,2)}_{ij}$ do not depend on $t$, so the interpolation weights are computed once, when the grid is generated, and stored in `Grid_data/<L>/`.
- **in $t$**, for the factor $\mathcal B$ whose first argument is $\theta_i t$ or $\phi t$, which lies between stored times. Here DYNAMITE uses cubic Hermite interpolation, with the time derivatives that the Runge–Kutta method provides anyway. The stored times change at every step, so the interpolation intervals are searched anew; because a step is small compared with $t$, the search starts from the previous result.

For each of the $L$ values of $\theta_i$, each integral is a sum over $L$ quadrature nodes, so one evaluation of the right-hand side costs $\mathcal O(L^2)$ operations, independent of the number of stored times. The time is dominated by the interpolations, which are limited by memory access rather than by arithmetic.

## One time step

1. Interpolate $C$ and $R$ at the nodes $\phi^{(1,2)}_{ij}$ and evaluate the memory integrals.
2. Evaluate the right-hand side of the equations of motion along the rays $\theta_i$.
3. Repeat for every stage of the Runge–Kutta method, and append the new time slice to the history.
4. Compare the Runge–Kutta error estimate with the tolerance $\epsilon$ and adapt the next step size. If the error exceeds $2\epsilon$, discard the last ten time slices and halve the step.
5. Every `--sparsify-interval` steps (100000 by default): remove time slices that the interpolation in $t$ does not need, and write a checkpoint.

## Cost

Each step has a fixed cost of $\mathcal O(L^2)$. As long as the step size grows with $t$, the run time grows sublinearly with the simulated time. Once the step size reaches the stability limit of the explicit Runge–Kutta method, the cost becomes linear in $t$. Without sparsification the memory grows linearly with the number of steps; with sparsification the growth is sublinear (about $t^{1/3}$ in the benchmarks of the [DYNAMITE paper](https://arxiv.org/abs/2604.06309)).

A related, earlier approach by Kim and Latz keeps a fixed number of equidistant times and, when the grid is full, removes every second point and doubles the step. It is very efficient, but the errors are not controlled, and for the mixed spherical $p$-spin model they grow rapidly. Choosing the $\theta_i$ and the time steps as described above avoids this.

## Error control

The following errors are controlled separately:

- the Runge–Kutta error, through the tolerance `-e` ([Time integration](time-integration.md));
- the interpolation and quadrature errors in $\theta$, through $L$ and the interpolation and spline orders chosen when generating the grid ([Grid and interpolation](interpolation-grids.md));
- the error from removing time slices, which is kept below the same tolerance `-e` ([Sparsification](sparsification.md)).

Because errors made early are carried along by the memory integrals, reaching late times requires small errors per step. The local criteria do not by themselves bound the error at late times, so results should be checked by repeating a run with a larger $L$ and a smaller `-e`.
