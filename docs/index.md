<section class="dmfe-hero" aria-labelledby="dmfe-hero-title">
	<div class="dmfe-hero__glow" aria-hidden="true"></div>
	<div class="dmfe-hero__grid">
		<div class="dmfe-hero__text">
			<!-- <p class="dmfe-hero__eyebrow">Open-source dynamical mean-field evolution</p> -->
			<h1 id="dmfe-hero-title">Long-time dynamics, lightning-fast performance.</h1>
			<p class="dmfe-hero__lede">
				DYNAMITE is an efficient solver of DMFT equations with long memory. It features sublinear cost, GPU support and reusable checkpoints.
			</p>
			<div class="dmfe-hero__cta">
				<a class="md-button md-button--primary" href="install/">Install</a>
				<a class="md-button" href="usage/">Browse usage</a>
				<a class="md-button" href="tutorials/first-run/">Try the first run</a>
			</div>
			<!-- <dl class="dmfe-hero__stats">
				<div>
					<dt>Scaling</dt>
					<dd>Sublinear in simulated time</dd>
				</div>
				<div>
					<dt>Backends</dt>
					<dd>Deterministic CPU + optional GPU</dd>
				</div>
				<div>
					<dt>Outputs</dt>
					<dd>Versioned HDF5 + checkpoints</dd>
				</div>
			</dl> -->
		</div>
		<!-- <div class="dmfe-hero__visual">
			<div class="dmfe-showcase">
				<p class="dmfe-showcase__label">Showcase lane</p>
				<div class="dmfe-showcase__cards">
					<div class="dmfe-showcase__card">
						<h3>CLI snapshots</h3>
						<p>Drop terminal captures or run summaries here when you are ready to highlight workflows.</p>
					</div>
					<div class="dmfe-showcase__card">
						<h3>Code embeds</h3>
						<p>Embed syntax-highlighted EOM snippets, grid specs, or screenshots without changing the layout.</p>
					</div>
				</div>
				<p class="dmfe-showcase__hint">This lane intentionally stays flexible for future examples or screenshots.</p>
			</div>
		</div> -->
	</div>
</section>

<section class="dmfe-section dmfe-terminal">
	<div class="dmfe-terminal-card">
		<div class="dmfe-terminal-card__glow" aria-hidden="true"></div>
		<!-- <div class="dmfe-terminal-card__body"> -->
			<div class="dmfe-terminal-card__header">
				<p class="dmfe-terminal-card__eyebrow">Terminal snapshot</p>
			</div>
			<!-- <div class="dmfe-terminal-card__player"> -->
```asciinema-player
{
	"file": "assets/demo.cast",
	"rows": 17,
	"cols": 102,
	"mkap_theme": "none",
	"auto_play": true,
	"loop": true
}
```
			<!-- </div> -->
		<!-- </div> -->
	</div>
</section>

<section class="dmfe-section">
	<div class="dmfe-section__header">
		<p class="dmfe-eyebrow">Why DYNAMITE</p>
		<h2>Purpose-built for aging and quench protocols</h2>
	</div>
	<div class="dmfe-feature-grid">
		<article class="dmfe-feature">
			<h3>Non-stationary solvers</h3>
			<p>Track two-time correlators C(t, t') and responses R(t, t') deep into aging, respecting the causal triangle.</p>
		</article>
		<article class="dmfe-feature">
			<h3>Interpolated memory</h3>
			<p>Two-dimensional sparse interpolation drops the asymptotic cost from O(T³) to sublinear while controlling error.</p>
		</article>
		<article class="dmfe-feature">
			<h3>Adaptive integrators</h3>
			<p>RK54 ⇄ SSPRK104 switching, stability-aware steps, and sparsification-aware tolerances keep runs safe.</p>
		</article>
		<article class="dmfe-feature">
			<h3>CPU baseline, GPU boosts</h3>
			<p>Deterministic CPU path plus optional CUDA kernels for acceleration after validation.</p>
		</article>
		<article class="dmfe-feature">
			<h3>Checkpoints + reproducibility</h3>
			<p>Versioned HDF5 outputs, restartable checkpoints, and pinned parameters for every trajectory.</p>
		</article>
		<article class="dmfe-feature">
			<h3>Plugin-style EOM modules</h3>
			<p>Drop in custom closures for new models via the EOM interface. Examples live in <code>concepts/eoms-and-observables.md</code>.</p>
		</article>
	</div>
</section>

## DMFT problem setting

We evolve correlation and response functions after a quench under closed dynamical equations of the form

$$
\partial_t C(t,t') = \mathcal{F}[C,R](t,t')\,,\qquad
\partial_t R(t,t') = \mathcal{G}[C,R](t,t')\,,\quad t\ge t'\,.
$$

The solver implements a numerical renormalization scheme with two-dimensional interpolation, reducing the asymptotic cost from cubic to sublinear in simulated time while controlling accuracy for aging observables.

### When DMFE is the right tool

- Your DMFT equations close on C and R with memory integrals over the past.
- You need long-time, high-accuracy trajectories (aging, quenches, or other non-stationary protocols).
- You rely on reproducible outputs, resumable checkpoints, and deterministic CPU references before enabling GPUs.

### Model assumptions

- Single-site, causal effective dynamics laid out on the triangular domain t ≥ t'.
- History terms expressible as integrals/convolutions evaluable on a 2D interpolation grid.
- Model-specific closures supplied via an EOM module (see `concepts/eoms-and-observables.md`).

## Quickstart

<div class="dmfe-quickstart">
	<div>
		<h3>Build</h3>
		<pre><code>./build.sh</code></pre>
	</div>
	<div>
		<h3>Short quench (L = 512)</h3>
		<pre><code>./RG-Evo -L 512 -l 0.5 -m 1e4 -D false</code></pre>
	</div>
	<div>
		<h3>Outputs</h3>
		<p>Versioned HDF5: <code>QKv</code>, <code>QRv</code>, <code>dQKv</code>, <code>dQRv</code>, <code>t1grid</code>, <code>rvec</code>, <code>drvec</code>. Binary + text summaries when HDF5 is unavailable.</p>
	</div>
</div>

## Typical workflow

<ol class="dmfe-workflow">
	<li><span>Pick or implement an EOM module defining the closures \( \mathcal{F}, \mathcal{G} \) for your model.</span></li>
	<li><span>Choose an interpolation grid (sizes/layout) that balances accuracy and runtime.</span></li>
	<li><span>Select an integrator and tolerances; adaptive RK54 is the usual starting point.</span></li>
	<li><span>Run a short trajectory, inspect stability + error estimates, and adjust grids/tolerances.</span></li>
	<li><span>Launch production runs with checkpoints so you can resume to extend time horizons.</span></li>
</ol>

## Outputs, performance, and accuracy

- Sparse 2D interpolation of history terms yields sublinear cost in simulated time for long runs.
- Stability-aware integrator switching maintains accuracy near stiff or transient regimes.
- Recommended starting grids live in `concepts/interpolation-grids.md`; refine per observable tolerances.
- CPU path is the reference; enable GPU when available for speed-ups after validating on your model.

## Cite DYNAMITE & get help

- Software: see **Reference → Cite** (powered by `CITATION.cff`).
- Method paper: J. Lang, S. Sachdev, S. Diehl, “Numerical renormalization of glassy dynamics,” arXiv:2504.06849.
- License: Apache-2.0 (see `LICENSE`).
- Support: follow `dev/testing.md` for issue templates or open a GitHub issue with your build info (compiler/CUDA, commit hash, minimal input).
