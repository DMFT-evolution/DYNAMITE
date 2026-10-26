# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/grid.svg" alt="Grid icon"/> Grid and interpolation

The fields are stored on the points $(t_n,\theta_i t_n)$, with a fixed grid in $\theta=t'/t$ and adaptive times $t_n$ (see [Algorithm](algorithm.md)). This page describes the $\theta$ grid, the interpolation in $\theta$ and in $t$, and the files in `Grid_data/<L>/`. How to generate grids is explained in [Generate grids](../tutorials/generate-grids.md) and the [Grid generator](../grid-generator.md) reference.

<figure style="text-align:center;">
	<img src="/DYNAMITE/assets/interpolation-grid-diagram.svg" alt="Grid on the triangle t' ≤ t" style="max-width: 680px; width: 100%; height: auto;" />
	<figcaption>Grid points on the triangle $0\le t'\le t$: adaptive in $t$, fixed in $\theta=t'/t$, dense near the diagonal and near $t'=0$.</figcaption>
</figure>

## The $\theta$ grid

For $i=1,\dots,L$ the grid generator uses

$$
\theta_i=\frac{\arctan\!\big(e^{\gamma}\big)-\arctan\!\big(e^{-\gamma\,y_i}\big)}{\arctan\!\big(e^{\gamma}\big)-\arctan\!\big(e^{-\gamma}\big)}\,,
\qquad y_i=\frac{2i-1-L}{L-1}\,,
\qquad \gamma=-\frac{L-1}{L}\,W_{-1}\!\Big(-\frac{1}{10\,t_{\max}}\Big)\,,
$$

where $W_{-1}$ is the lower branch of the Lambert $W$ function and $t_{\max}$ the largest time one intends to simulate (option `-M` of the grid generator). This is the family of grids of Eq. (16) of the [DYNAMITE paper](https://arxiv.org/abs/2604.06309) (arXiv version 1), where the expression for $\gamma$ differs slightly. The grid has $\theta_1=0$ and $\theta_L=1$, and near both ends it is geometric: $\theta_i$ and $1-\theta_i$ change by a constant factor $e^{-2\gamma/(L-1)}$ per point. The number of points that resolve a fixed microscopic window near either end therefore decreases only logarithmically with $t$.

**Choice of $t_{\max}$.** $t_{\max}$ should not be smaller than the latest time you simulate; otherwise the resolution of the microscopic scales degrades at late times. The grid shipped in `Grid_data/512/` uses $t_{\max}=10^5$. For longer runs, generate a grid with a larger `-M` (see [Generate grids](../tutorials/generate-grids.md)).

**Optional reshaping.** The generator can evaluate the formula above at a remapped index, $i\to\alpha\,\varphi_\delta(i)+(1-\alpha)\,i$ with

$$
\varphi_\delta(i)=\frac{L-1}{2}\left(\frac{g_\delta(y_i)}{g_\delta(1)}+1\right)+1\,,\qquad
g_\delta(x)=\operatorname{sign}(x)\Big[\big(|x|^3+\delta^3\big)^{1/3}-\delta\Big]\,,
$$

controlled by `--alpha` $\in[0,1]$ and `--delta` $\ge0$. This shifts points from the two edges towards the middle of the $\theta$ range, where $t'\gg\tau_{\rm micro}$ and $t-t'\gg\tau_{\rm micro}$, and keeps the end points. The default $\alpha=\delta=0$ gives the grid above.

**Other grids.** Any strictly increasing set of nodes with $\theta_1=0$ and $\theta_L=1$ can be supplied with `--theta-file`. The method does not depend on the particular grid; what matters is that it resolves the microscopic scales near both ends. Depending on the model, more aggressive grids with fewer points can reach the same accuracy, but in our experience the gain in run time is small (below about 50%); the default grid is deliberately conservative.

## Interpolation in $\theta$

The memory integrals need the fields at the points $\phi^{(1)}_{ij}=\theta_i\theta_j$ and $\phi^{(2)}_{ij}=\theta_i+(1-\theta_i)\theta_j$, and, for the integral over $(t',t)$, at the ratio $\theta_i/\phi^{(2)}_{ij}$. None of these lie on the grid, so they are obtained by interpolation with weights that are computed once by the grid generator. The available methods (option `--interp-method`) are

| Method | Interpolation |
|---|---|
| `index-poly` (default) | Lagrange polynomial in the index $i$, in barycentric form |
| `index-rational` | Floater–Hormann rational interpolation in the index |
| `index-hermite` | cubic Hermite in the index |
| `index-bspline` | B-spline kernel in the index (smoothing, not interpolating for degree ≥ 2) |
| `poly`, `rational`, `bspline` | the same, directly in $\theta$ |

The `index-` methods interpolate the fields as functions of the grid index, $\mathcal A(t,\theta(i))$, which is smooth because $\theta(i)$ is a smooth function. They are more accurate and, above all, more stable than interpolation in $\theta$ on the strongly non-uniform grid, and they are the default. The order (`--interp-order`, default 9) sets the size of the stencil, $n+1$ points. Higher orders are more accurate for smooth data, but are slower and amplify noise more strongly; for the default settings, the interpolation error is far below the Runge–Kutta error.

## Quadrature

The integrals over $u$ (see [Algorithm](algorithm.md)) are evaluated as $\sum_j\nu_j\,(\dots)$ with weights $\nu_j$ obtained by integrating a spline through the grid points. The spline order is set with `--spline-order` (default 5, i.e. quintic). By default (`--int-method index-spline`) the spline is built in the index variable and integrated using the known derivative $\theta'(i)$; `theta-spline` builds it directly in $\theta$. The weights are stored in `int.dat`.

## Interpolation in $t$

Values at times between stored slices are obtained by cubic Hermite interpolation from the two neighbouring slices, using the stored derivatives `dQKv`, `dQRv`. The error of this interpolation, integrated over a step, scales like $\Delta t^5$, comparable to the Runge–Kutta error. With `-R true` the response is interpolated as $\log R$ instead; this is useful where $R$ decays exponentially, and the code falls back to $R$ wherever a stencil contains non-positive values.

## Files in `Grid_data/<L>/`

| File | Size | Content |
|---|---|---|
| `theta.dat` | $L$ | the nodes $\theta_i$ |
| `phi1.dat`, `phi2.dat` | $L\times L$ | the points $\phi^{(1)}_{ij}$ and $\phi^{(2)}_{ij}$ |
| `int.dat` | $L$ | quadrature weights $\nu_j$ |
| `posA1y.dat`, `posA2y.dat`, `posB2y.dat` | $L\times L$ | positions, as fractional grid index, of $\phi^{(1)}_{ij}$, $\phi^{(2)}_{ij}$ and $\theta_i/\phi^{(2)}_{ij}$ |
| `indsA1y.dat`, `indsA2y.dat`, `indsB2y.dat` | $L\times L$ | first grid index of the interpolation stencil for each of these points |
| `weightsA1y.dat`, `weightsA2y.dat`, `weightsB2y.dat` | $L\times L\times(n+1)$ | interpolation weights |
| `grid_params.txt` | | parameters and command line used to generate the grid |

The program reads these files at start-up; nothing about the $\theta$ grid is computed during a run. The parameters in `grid_params.txt` are copied into the `params.txt` of every run.

## Choosing $L$

The shipped grid has $L=512$. Larger $L$ is more accurate and more expensive: the work per step grows as $L^2$, and so does the size of the grid files. The memory needed for the history grows as $L$ times the number of stored slices. Check that the observables you need do not change when $L$ is doubled, in particular near dynamical transitions.
