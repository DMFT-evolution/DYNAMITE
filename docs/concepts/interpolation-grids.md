# <img class="icon icon-lg icon-primary" src="/DYNAMITE/assets/icons/grid.svg" alt="Grid icon"/> Interpolation grids (paper-defined, non‑equidistant)

The *interpolation grid* is one of DYNAMITE’s core design choices: it fixes **where** the two‑time Green’s functions are sampled on the triangular domain $t'\le t$, and it ships with the **quadrature weights, interpolation stencils, and index maps** used to evaluate memory integrals efficiently.

Conceptually, the algorithm keeps a *fixed* number of samples in the dimensionless time ratio
$\theta=t'/t\in[0,1]$ and evolves only the outer time $t$ with an adaptive ODE solver. The grid is chosen so that microscopic features near $t'\approx t$ (small relative time) and near $t'\approx 0$ (quench/onset region) remain resolved for all simulated ages $t$.

This page explains (i) the basic idea and why it matters for accuracy/scaling, (ii) what’s stored under `Grid_data/<L>/`, and (iii) the paper equations used to generate the shipped grid.

## Quick orientation

### What is being gridded?

DYNAMITE parameterizes the triangular domain using the time ratio

\[
	heta = t'/t \in [0,1],\quad \text{with } t'\le t.
\]

The grid is a **fixed set of monotone nodes** $\{\theta_i\}_{i=1}^N$ that is intentionally dense near $\theta\approx 0$ and $\theta\approx 1$.

### Why non‑equidistant?

At long times, Green’s functions can retain sharp structure for small relative times $\tau=t-t'\lesssim\tau_{\rm micro}$ (i.e. very close to the diagonal $t'\approx t$), and (after a quench) can also remain sensitive to the early‑time region $t'\lesssim\tau_{\rm micro}$. The paper’s key requirement is that the grid spacings be fine enough to resolve these microscopic scales *when mapped back to physical time*, while keeping the number of ratio nodes fixed.

In the notation of the paper, this is captured by the resolution condition (main text Eq. (2)):

\[
	heta_{i+1}-\theta_i \ll \frac{\tau_{\rm micro}}{t}\quad \text{for all } i \text{ with } \min(\theta_i,1-\theta_i)\lesssim \frac{\tau_{\rm micro}}{\min(t',\tau)}.
\]

Practically, this is why the grid must be dense near $\theta=0$ and $\theta=1$.

### Where does it show up in practice?

- Runtime interpolation uses precomputed **stencils** (indices + weights) derived from this fixed $\theta$ grid.
- Memory integrals use spline‑consistent **quadrature weights** on the same nodes.
- All of these are bundled under `Grid_data/<L>/` and loaded by the code at startup.

<figure style="text-align:center;">
	<img src="/DYNAMITE/assets/interpolation-grid-diagram.svg" alt="Triangular time domain with dense and sparse grid regions" style="max-width: 680px; width: 100%; height: auto;" />
	<figcaption>Triangular time domain with dense and sparse grid regions.</figcaption>
	
</figure>

Only the triangular domain $t' \le t$ is simulated. The algorithm samples functions as $\mathcal A(t,\theta)\equiv A(t,\theta t)$ on a fixed irregular $\theta$ grid, and (crucially) evaluates memory integrals by interpolating along derived contours that remain dense in the RG‑relevant regions.

DYNAMITE uses exactly the non‑equidistant, nested time grid defined in Lang–Sachdev–Diehl (Phys. Rev. Lett. **135**, 247101 (2025), [doi:10.1103/z64g-nqs6](https://journals.aps.org/prl/abstract/10.1103/z64g-nqs6)). All node locations and quadrature/interpolation metadata are precomputed and shipped under `Grid_data/<L>/` for $L\in\{512,1024,2048\}$.

On this page:

- **Concept and code artifacts:** what the shipped grid package contains and how runtime uses it.
- **Exact definition:** the explicit paper equations that define the base grid.
- **Optional generator feature:** the \(\alpha/\delta\) index remapping (defaults reproduce the paper grid exactly).
- **Practical use:** choosing $L$, regeneration, and quick convergence checks.

> **Important:** treat the grid choice as an algorithmic ingredient, not a cosmetic discretization. Using an equidistant grid often destroys the scaling and can increase cost dramatically.

## What the files contain (and how the runtime uses them)

The grid package under `Grid_data/<L>/` is meant to be *complete*: the code should not have to recompute interpolation topology or quadrature rules at runtime.

- `theta.dat`  — the fixed monotone node set $\{\theta_i\}$ for $\theta=t'/t$.
- `phi1.dat`, `phi2.dat` — the two contour families used to discretize the memory integrals in the paper (main text Eq. (3)), evaluated from the same base nodes:
	- $\phi^{(1)}_{ij}=\theta_i\,\theta_j$ (dense near the onset / early times)
	- $\phi^{(2)}_{ij}=\theta_j+(1-\theta_j)\,\theta_i$ (dense near the diagonal / small relative time)
- `int.dat` — quadrature weights for integration over the irregular node set (open‑clamped B‑spline; default $s=5$).

> Notation note: the paper uses $\theta=t'/t$ for the time ratio. In data files, the node list is stored in `theta.dat`. The auxiliary contour coordinates are stored as `phi1.dat` and `phi2.dat`.

Interpolation metadata (for fast gathers during convolution / stencil evaluation):

- `posA1y.dat`, `posA2y.dat`, `posB2y.dat` — physical positions of the interpolation stencils assembled from $\theta$, $\varphi^{(1)}$, $\varphi^{(2)}$.
- `indsA1y.dat`, `indsA2y.dat`, `indsB2y.dat` — index maps into base arrays (e.g. $C$, $R$, and derivatives) for fast gathers.
- `weightsA1y.dat`, `weightsA2y.dat`, `weightsB2y.dat` — interpolation weights matching the index maps.

Provenance:

- `grid_params.txt` — generator parameters (len, $T_\max$, spline order, interpolation method/order, FH window if rational, optional `alpha`/`delta`, and the command line used).

## Generating/updating the grids and metadata

You can (re)generate a grid package via the built-in CLI:

```bash
./RG-Evo grid --len L --Tmax 100000 --dir L \
			  --interp-method poly --interp-order 9
# Short aliases: -L, -M, -d, -V, -s, -m, -o, -f
```

Flags:

- `--len L` selects the grid length (N=L).
- `--Tmax X` sets the long-time scale used by the $\theta$ mapping.
- `--dir SUBDIR` chooses the output subdirectory under `Grid_data/` (default: the value of `L`).
- `--alpha X` in [0,1] and `--delta X \ge 0` enable the optional smooth index remapping for $\theta$ (defaults 0, i.e. paper‑exact). The chosen values are recorded in `grid_params.txt`.
- `--spline-order n` controls the integration (quadrature) B‑spline degree (default: 5).
- `--interp-method {poly|rational|bspline}` chooses the interpolation method for metadata.
- `--interp-order n` sets interpolation degree/order.
- `--fh-stencil m` (rational only) sets the Floater–Hormann window size $m\ge n+1$ (default $n+1$).

Method notes:

- `poly` (barycentric Lagrange) produces local stencils with exactly $n+1$ weights per entry; minimal per-evaluation cost.
- `rational` (Floater–Hormann) with default $m=n+1$ matches `poly`; increasing `--fh-stencil` to $n+3\dots n+7$ blends overlapping local stencils and improves robustness on highly non-uniform grids while staying local (exactly $m$ weights per entry).
- `bspline` produces a dense global map from base samples to outputs; appropriate when you need spline smoothness/derivatives and can reuse the same data across many evaluations.

## Explicit equations (as in Phys. Rev. Lett. **135**, 247101 (2025))

We parametrize the two‑point functions on the triangular domain $t_2 \le t_1$ by the time ratio $\theta = t_2/t_1 \in [0,1]$, i.e.

\[
\mathcal G(t_1,\theta) \equiv G(t_1,\theta\,t_1).
\]

The fixed irregular grid $\{\theta_k\}_{k=1}^L$ (dense near $0$ and $1$) is given by the supplemental Eq. (S1):

\[
\xi_k = \frac{2k - 1 - L}{L - 1},\quad k=1,\dots,L,\qquad
\sigma_0 = -\,W\!\left(-\frac{1}{t_{\max}}\right),
\]

and the monotone arctan‑based mapping

\[
	heta_k = \frac{\arctan(\sigma_\infty) - \arctan(\sigma_\infty - \sigma_0\,\xi_k)}{\arctan(\sigma_\infty) - \arctan(\sigma_\infty - \sigma_0)}.
\]

Here $\sigma_\infty$ is a large positive constant that fixes end‑point crowding; any choice that yields sufficient density near $0$ and $1$ for your $t_{\max}$ is acceptable. This mapping is analytically invertible and yields near‑endpoint spacings that satisfy the resolution condition (paper Eq. (2)).

## Optional index remapping (alpha/delta)

In addition to the paper‑exact grid above, the implementation supports an **optional** smooth non‑linear remapping of the *fractional index* prior to evaluating the underlying $\Theta(x)$ mapping (i.e. before producing node locations in $\theta\in[0,1]$).

- `alpha` ∈ [0,1]: blends toward the non‑linear map.
- `delta \ge 0`: controls the “softness” of the remapping.

The defaults `alpha=0, delta=0` reproduce the paper grid exactly. Non‑zero `alpha` re‑distributes nodes while preserving monotonicity and endpoints. If used, values are recorded in `Grid_data/<subdir>/grid_params.txt`.

### Closed form of the remapping

Let $L$ be the grid length and let $x\in[1, L]$ denote the 1‑based fractional index at which the original paper mapping $\Theta(x)$ (the function that produces $\theta\in[0,1]$) is evaluated. Define

\[
\begin{aligned}
s(x) &\equiv \frac{2x - L - 1}{L - 1}\in[-1,1],\\
c &\equiv \frac{L - 1}{2},\\
g_\delta(s) &\equiv \operatorname{sign}(s)\left[(|s|^3 + \delta^3)^{1/3} - \delta\right],\\
g_1 &\equiv (1 + \delta^3)^{1/3} - \delta,\\
\varphi(x;\delta) &\equiv c\left( \frac{g_\delta(s(x))}{g_1} + 1 \right) + 1.
\end{aligned}
\]

With a blend parameter $\alpha\in[0,1]$ and softness $\delta\ge 0$, the remapped index is

\[
x_\alpha \;\equiv\; \alpha\,\varphi(x;\delta) + (1-\alpha)\,x,
\]

and the modified grid is obtained by composition with the original mapping:

\[
\Theta_{\alpha,\delta}(x) \;\equiv\; \Theta\!\big(\, \alpha\,\varphi(x;\delta) + (1-\alpha)\,x \,\big).
\]

Remarks:
- $\alpha=0$ (any $\delta$) yields the identity in index space, i.e., the paper‑exact grid. $\alpha=1$ applies the full non‑linear remapping.
- The normalization by $g_1$ clamps the transform so that endpoints map to endpoints (monotone, range preserved).

For memory integrals (paper Eq. (3)), the paper introduces an integration variable $\phi$ and uses two contour families on the same $\theta$ grid:

\[
\rho_k^{(1)}(\alpha_j) = \phi_k\,\alpha_j,\qquad
\rho_k^{(2)}(\alpha_j) = \alpha_j + (1-\alpha_j)\,\phi_k,\qquad
\alpha_j \equiv \phi_j.
\]

These generate the “mixed” directions required by the convolution structure and define the 2D interpolation stencils used at runtime.

DYNAMITE uses the following notation in code and data:

- $\{\theta_i\}$ (stored in `theta.dat`).
- The two induced contour families (stored in `phi1.dat` and `phi2.dat`).
- Spline‑consistent quadrature weights for the irregular grid (stored in `int.dat`, B‑spline degree $s$; default $s=5$).

## Performance note

Using these paper‑defined non‑equidistant grids is the primary reason the method attains sublinear growth of cost with simulated time. Changing to an equidistant grid will typically increase complexity toward the cubic baseline and should be avoided.

## Choosing L and checking convergence

- Available L: 512 / 1024 / 2048. Larger L → higher fidelity at higher cost.
- Validate by comparing observables (energy(t), C(t,t), C(t, α t)) across L.
- Near phase transitions, prefer larger L to capture critical scaling.

Inputs directory structure:

- `Grid_data/<L>/theta.dat`, `phi1.dat`, `phi2.dat`, `int.dat`
- `Grid_data/<L>/posA1y.dat`, `posA2y.dat`, `posB2y.dat`
- `Grid_data/<L>/indsA1y.dat`, `indsA2y.dat`, `indsB2y.dat`
- `Grid_data/<L>/weightsA1y.dat`, `weightsA2y.dat`, `weightsB2y.dat`

## Conceptual figure: triangular time domain and non‑equidistant grid

The figure above is the key picture to keep in mind: dense near $(t,t')\approx(0,0)$, dense near the diagonal at late times, and sparse in between.
