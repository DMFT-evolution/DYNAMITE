<section class="dmfe-hero" aria-labelledby="dmfe-hero-title">
	<div class="dmfe-hero__glow" aria-hidden="true"></div>
	<div class="dmfe-hero__grid">
		<div class="dmfe-hero__text">
			<h1 id="dmfe-hero-title">Dynamical mean-field equations to long times</h1>
			<p class="dmfe-hero__lede">
				DYNAMITE solves the two-time dynamical mean-field equations (DMFE) of the mixed spherical <em>p</em>-spin model after a quench, up to times of order 10<sup>7</sup>. It runs on an ordinary CPU and, much faster, on an NVIDIA GPU.
			</p>
			<div class="dmfe-hero__cta">
				<a class="md-button md-button--primary" href="tutorials/first-run/">First run</a>
				<a class="md-button" href="install/">Install</a>
				<a class="md-button" href="usage/">Usage</a>
			</div>
		</div>
	</div>
</section>

<section class="dmfe-section dmfe-terminal">
	<div class="dmfe-terminal-card">
		<div class="dmfe-terminal-card__glow" aria-hidden="true"></div>
			<div class="dmfe-terminal-card__header">
				<p class="dmfe-terminal-card__eyebrow">A short run</p>
			</div>
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
	</div>
</section>

<section class="dmfe-section">
	<div class="dmfe-section__header">
		<p class="dmfe-eyebrow">Overview</p>
		<h2>What the code does</h2>
	</div>
	<div class="dmfe-feature-grid">
		<article class="dmfe-feature">
			<h3>Two-time functions</h3>
			<p>Evolves the correlation C(t,t') and the response R(t,t') after a quench, including the aging regime.</p>
		</article>
		<article class="dmfe-feature">
			<h3>Fixed grid in t'/t</h3>
			<p>The second time is sampled on a fixed non-uniform grid in θ = t'/t, so every time step has the same cost, however long the history.</p>
		</article>
		<article class="dmfe-feature">
			<h3>Adaptive Runge–Kutta</h3>
			<p>Dormand–Prince at short times, SSPRK(10,4) later. The step size follows an embedded error estimate.</p>
		</article>
		<article class="dmfe-feature">
			<h3>CPU or GPU</h3>
			<p>Builds with any C++17 compiler. A CUDA GPU shortens long runs considerably, but is not required.</p>
		</article>
		<article class="dmfe-feature">
			<h3>Checkpoints</h3>
			<p>The state is saved periodically. Restarting with the same parameters continues the run.</p>
		</article>
	</div>
</section>

## The equations

DYNAMITE integrates causal integro-differential equations for $C(t,t')$ and $R(t,t')$ on the triangle $0\le t'\le t$,

$$
\begin{aligned}
\partial_t C(t,t') &= -\mu(t)\,C(t,t') + \int_0^t \mathrm ds\, K(t,s)\,C(s,t') + \int_0^{t'} \mathrm ds\, D(t,s)\,R(t',s) + F_C(t,t')\,,\\
\partial_t R(t,t') &= -\mu(t)\,R(t,t') + \delta(t-t') + \int_{t'}^t \mathrm ds\, K(t,s)\,R(s,t')\,,
\end{aligned}
$$

where the memory kernels $K$ and $D$ depend on $C$ and $R$. With a uniform time step, reaching a time $t$ costs $\mathcal O(t^3)$ operations and $\mathcal O(t^2)$ memory. DYNAMITE instead stores the fields on a fixed non-uniform grid in $\theta=t'/t$ and an adaptive grid in $t$, evaluates the memory integrals by interpolation with precomputed weights, and removes stored time slices that the interpolation does not need. The run time then grows linearly with $t$ at late times, and the memory sublinearly. The method is summarized in [Algorithm](concepts/algorithm.md) and described in detail in the [DYNAMITE paper](https://arxiv.org/abs/2604.06309).

The released program contains the equations of the mixed spherical $p$-spin model, $f(q)=\lambda q^p+(1-\lambda)q^s$, quenched from equilibrium at temperature $T_0$ to a bath at temperature $T$ ([Equations](concepts/eoms-and-observables.md)). Other models of the same form require changes to a few model-specific source files ([Adapting the code to another model](concepts/other-models.md)).

## Quick start

<div class="dmfe-quickstart">
	<div>
		<h3>Build</h3>
		<pre><code>./build.sh</code></pre>
	</div>
	<div>
		<h3>Run (CPU, a few minutes)</h3>
		<pre><code>./RG-Evo -p 3 -q 4 -l 0.5 -t 10 -g false -o Results/</code></pre>
	</div>
	<div>
		<h3>Look at the result</h3>
		<p><code>correlation.txt</code> in the run directory lists the time <i>t</i>, the energy <i>E(t)</i> and <i>C(t,0)</i>.</p>
	</div>
</div>

## Typical workflow

<ol class="dmfe-workflow">
	<li><span>Choose the model (<code>-p</code>, <code>-q</code>, <code>-l</code>) and the two temperatures (<code>-T</code>, <code>-G</code>).</span></li>
	<li><span>Start with the shipped grid <code>L = 512</code> and the default tolerance <code>-e 1e-10</code>.</span></li>
	<li><span>Run to a moderate time and check the energy and <i>C(t,0)</i>.</span></li>
	<li><span>Repeat with a larger <code>L</code> and a smaller <code>-e</code> to check that the observables you need have converged.</span></li>
	<li><span>Launch the long run. It writes checkpoints and can be extended later by restarting it with a larger <code>-t</code>.</span></li>
</ol>

## Citing and help

- The code and the method are described in J. Lang, V. Citro, L. Leuzzi, F. Ricci-Tersenghi, “DYNAMITE: A high-performance framework for solving Dynamical Mean-Field Equations,” [arXiv:2604.06309](https://arxiv.org/abs/2604.06309) (under review). Please cite this paper when you use DYNAMITE; see [Cite](reference/cite.md).
- License: Apache-2.0 (see [`LICENSE`](https://github.com/DMFT-evolution/DYNAMITE/blob/main/LICENSE)).
- Questions and bug reports: open an issue on [GitHub](https://github.com/DMFT-evolution/DYNAMITE/issues) and include the `params.txt` of the run.
