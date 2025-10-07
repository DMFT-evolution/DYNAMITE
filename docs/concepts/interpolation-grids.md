# <img class="icon icon-lg icon-primary" src="/DMFE/assets/icons/grid.svg" alt="Grid icon"/> Interpolation grids (paper-defined, non‑equidistant)

DMFE uses exactly the non‑equidistant, nested time grid defined in Lang–Sachdev–Diehl (arXiv:2504.06849). The grid is multi‑scale and highly non‑uniform by design to resolve short‑time singular structure and long‑time aging simultaneously. All node locations and quadrature data are precomputed and shipped under `Grid_data/<L>/` for L ∈ {512, 1024, 2048}.

Why this matters: The algorithm’s sublinear scaling depends critically on this grid. Although not extremely sensitive to tiny details, using a highly non‑equidistant grid with nested blocks is essential; equidistant grids defeat the renormalization gains and dramatically increase cost.

## Explicit equations (as in arXiv:2504.06849)

We parametrize the two‑point functions on the triangular domain $t_2 \le t_1$ by the time ratio $\phi = t_2/t_1 \in [0,1]$, i.e.

\[
\mathcal G(t_1,\phi) \equiv G(t_1,\phi\,t_1).
\]

The fixed irregular grid $\{\phi_k\}_{k=1}^L$ (dense near $0$ and $1$) is given by the supplemental Eq. (S1):

\[
\xi_k = \frac{2k - 1 - L}{L - 1},\quad k=1,\dots,L,\qquad
\sigma_0 = -\,W\!\left(-\frac{1}{t_{\max}}\right),
\]

and the monotone arctan‑based mapping

\[
\phi_k = \frac{\arctan(\sigma_\infty) - \arctan(\sigma_\infty - \sigma_0\,\xi_k)}{\arctan(\sigma_\infty) - \arctan(\sigma_\infty - \sigma_0)}.
\]

Here $\sigma_\infty$ is a large positive constant that fixes end‑point crowding; any choice that yields sufficient density near $0$ and $1$ for your $t_{\max}$ is acceptable. This mapping is analytically invertible and yields near‑endpoint spacings that satisfy the resolution condition (paper Eq. (2)).

For memory integrals (paper Eq. (3)), two contour families on the same grid are used:

\[
\rho_k^{(1)}(\alpha_j) = \phi_k\,\alpha_j,\qquad
\rho_k^{(2)}(\alpha_j) = \alpha_j + (1-\alpha_j)\,\phi_k,\qquad
\alpha_j \equiv \phi_j.
\]

These generate the “mixed” directions required by the convolution structure and define the 2D interpolation stencils used at runtime.

DMFE uses the following notation in code and data:

- $\theta \equiv \{\phi_k\}$ (stored in `theta.dat`).
- $\varphi^{(1)} \equiv \{\rho_k^{(1)}(\alpha_j)\}$ (stored in `phi1.dat`).
- $\varphi^{(2)} \equiv \{\rho_k^{(2)}(\alpha_j)\}$ (stored in `phi2.dat`).
- Quintic‑spline quadrature weights for the irregular grid (stored in `int.dat`).

## What the files contain

- `theta.dat`  — primary (monotone) parameterization θ of time‑ratio nodes {φ_k}.
- `phi1.dat`, `phi2.dat` — the two contour families ϕ¹, ϕ² evaluated on the φ‑grid; used to assemble 2D interpolation stencils consistent with the renormalized layering.
- `int.dat` — quadrature weights associated with {φ_k} for accurate evaluation of memory integrals.
- `posA1y.dat`, `posA2y.dat`, `posB2y.dat` — physical positions of interpolation stencils assembled from θ, ϕ¹, ϕ² (A/B families correspond to distinct directional stencils).
- `indsA1y.dat`, `indsA2y.dat`, `indsB2y.dat` — index maps into the base arrays (C, R, and their derivatives) for fast gathers.
- `weightsA1y.dat`, `weightsA2y.dat`, `weightsB2y.dat` — interpolation weights matching the index maps.

These datasets jointly define the exact nodes and interpolation/quadrature rules used at runtime; they are identical to those used in the paper’s benchmarks.

### Generating/updating the grids and metadata

You can (re)generate a grid package via the built-in CLI:

```bash
./RG-Evo grid --len L --Tmax 100000 --dir L \
			  --interp-method poly --interp-order 9
# Short aliases: -L, -M, -d, -V, -s, -m, -o, -f
```

Flags:
- `--len L` selects the grid length (N=L).
- `--Tmax X` sets the long-time scale used by the θ mapping.
- `--dir SUBDIR` chooses the output subdirectory under `Grid_data/` (default: the value of L).
- `--spline-order n` controls the integration (quadrature) spline order.
- `--interp-method {poly|rational|bspline}` chooses the interpolation method for metadata.
- `--interp-order n` sets interpolation degree/order.
 - `--fh-stencil m` (rational only) sets the Floater–Hormann window size m ≥ n+1 (default n+1). Wider m blends multiple degree-n local stencils for additional stability on irregular grids while keeping locality.

Method notes:
- `poly` (barycentric Lagrange) produces local stencils with exactly n+1 weights per entry. This minimizes per-evaluation cost when the interpolated data change often.
- `rational` (Floater–Hormann) with default m=n+1 matches `poly`; increasing `--fh-stencil m` to n+3..n+7 blends overlapping local stencils and improves robustness on highly non-uniform grids while staying local (exactly m weights per entry).
- `bspline` produces a dense global map from base samples to outputs, appropriate when you need spline smoothness/derivatives and can reuse the same data across many evaluations.

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

<figure style="text-align:center;">
	<img src="../../assets/interpolation-grid-diagram.svg" alt="Triangular time domain with dense and sparse grid regions" style="max-width: 680px; width: 100%; height: auto;" />
	<figcaption>Triangular time domain with dense and sparse grid regions.</figcaption>
  
</figure>

Only the triangular domain t' ≤ t is simulated; points are dense at both short times (near the origin) and along the diagonal at long times (where the correlation length R grows like R(τ) with τ = t - t'), with a sparse intermediate region where R ~ 1/t. This multi-scale structure reflects the renormalized layering that enables sublinear algorithmic scaling.
