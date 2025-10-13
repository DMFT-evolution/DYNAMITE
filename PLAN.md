# Visual appeal and scientific presentation plan

Audience: PhD-level researchers in theoretical physics, mathematics, and CS. Goal: keep a sober, professional tone while adding clear visual anchors and navigational affordances. All assets below are generated and committed under `docs/assets/` and can be used immediately.

## Immediate, low-risk changes (ready now)

- Site branding
  - Set logo and favicon to `docs/assets/logo-dmfe.svg` (wired in `mkdocs.yml`).
  - Keep palette as-is to match Material defaults; logo uses harmonious indigo/blue gradient.

- Home hero and diagrams
  - Embed method pipeline and architecture SVGs on the homepage:
    - `docs/assets/diagram-method.svg`
    - `docs/assets/diagram-architecture.svg`
  - Both are vector, light/dark friendly, and sized responsively by the theme.

- Gentle iconography for expert scanning
  - Use Material icon shortcodes sparingly in markdown (not inside raw HTML): `:material-cpu-64-bit:`, `:material-nvidia:`, `:material-function:`. Avoid overuse; keep to headings or callouts.

- Callouts for key claims
  - Use admonitions to highlight guarantees and caveats, for example:
    - `!!! note "Stability switching"` One-liner on RK54 → SSPRK104 with thresholds.

## Short-term additions (1–2 commits)

- Insert small module badges in Concepts pages (optional):
  - EOMs: `:material-function:`
  - Interpolation grids: `:material-grid:`
  - Sparsification: `:material-content-cut:`
  - Convolution: `:material-chart-bell-curve:`

- Add a compact figure to `docs/concepts/interpolation-grids.md` (already present: `docs/assets/interpolation-grid-diagram.svg`). Add a caption explaining node layout and causal triangle.

- Add a “Try it” panel on `usage.md` with a minimal parameter set and expected outputs.

## Medium-term ideas (implement incrementally)

- Visuals
  - Add a small sparkline-style plot of typical aging observable trajectories (C(t,t'), R(t,t')). For now, we can generate neutral SVG polylines approximating decays; later swap with vetted figures.

- Theming refinements
  - Consider Material features to aid long pages: `navigation.indexes`, `toc.integrate`.
  - Keep typography to system fonts to avoid external fetches.

## Concrete tasks and file hooks

1) Logo/Favicon
   - Done: `mkdocs.yml` sets `logo:` and `favicon:` to `assets/logo-dmfe.svg`.

2) Homepage
   - Done: Hero block and figures in `docs/index.md`.

3) Concepts
   - Action: Insert icon badges and figure references in:
     - `docs/concepts/interpolation-grids.md` (uses `assets/interpolation-grid-diagram.svg`).
     - `docs/concepts/architecture.md` (reference `assets/diagram-architecture.svg`).

4) Usage
   - Action: Add a `!!! example` callout with a minimal run and expected files.

5) Optional next visuals I can generate on request
   - A compact SVG emphasizing causal triangle geometry with sample nodes.
   - Minimalistic plots of C(t,t') and R(t,t') as stylized contours or line cuts.

## Acceptance criteria

- Docs build locally without extra dependencies beyond MkDocs + Material.
- Homepage shows the new logo, pipeline, and architecture figures; readable in light/dark modes.
- No JS bundling or theme overrides required; SVGs scale and include alt text.

## Notes

- All generated assets are vector-only, with neutral colors suitable for scientific audiences.
- When you supply real figures, we can swap them without altering structure or captions.

## Documentation maturity roadmap (closing gaps)

Goal: bring the documentation to a “research toolkit” standard that is comprehensive, reproducible, and maintainable. The focus is on API completeness, versioning, reproducible examples, and support materials.

### Objectives

- Publish a navigable API reference generated from the source (headers) and ship it on the site.
- Provide versioned documentation with a visible version switcher and per-release snapshots.
- Add a “Cite” page and repository citation metadata (DOI-backed) to enable proper academic referencing.
- Create a compact, executable examples gallery using notebooks with cloud execution badges.
- Centralize troubleshooting with an FAQ page; add a performance/benchmarking guide.
- Offer a complete CLI reference (flags, defaults, examples) under Reference.
- Surface changelog and roadmap within the site navigation.
- Add community/support and API stability/deprecation policy pages.
- Improve Doxygen signal (grouping, brief docs, diagrams) and harden the docs CI pipeline.

### Phase 1 — Core reference and site structure (1–2 PRs)

1) API reference
  - Ensure `docs/gen_api.sh` runs Doxygen and Doxybook2 to generate markdown into `docs/reference/api/`.
  - CI: install Doxybook2 and keep Graphviz/Doxygen (update `.github/workflows/docs.yml`).
  - MkDocs: include the generated `reference/api/` in navigation (avoid excluding it) and add a landing page.

2) Essential pages (create stubs if content is short initially)
  - `docs/howto/faq.md`: common installation/runtime issues and fixes.
  - `docs/concepts/performance.md`: CPU/GPU tuning, memory limits, profiling tips, scaling notes.
  - `docs/reference/changelog.md`: link through to repository `CHANGELOG.md`.
  - `docs/dev/api-stability.md`: what is stable/experimental; deprecation policy.
  - `docs/community/support.md`: how to get help, file issues, security contact.

3) Navigation updates
  - Add the above pages under appropriate groups (How-to, Concepts, Reference, Developer).
  - Keep URLs stable; use human-readable titles.

Acceptance criteria (Phase 1)
- Docs CI installs Doxybook2 and builds the API pages without warnings; `mkdocs build --strict` passes.
- Site navigation exposes “C++ API” under Reference; pages render with search working.
- FAQ, Performance, Changelog, API Stability, and Support pages exist and are reachable.

### Phase 2 — Reproducible examples and citation (1 sprint)

1) Executable examples gallery
  - Add minimal notebooks demonstrating: first run (CPU-only small grid), reading outputs, basic plotting.
  - Add Binder/Colab badges; provide `requirements.txt` or `environment.yml` for reproducibility.
  - Include or generate a tiny example dataset to keep runtime under a minute.

2) Citation and repository metadata
  - Add `docs/concepts/citation.md` with canonical references and BibTeX.
  - Add `CITATION.cff` to the repo root; integrate releases with a DOI provider (e.g., via GitHub + archiving service).
  - Surface “Cite” in the site navigation/footer.

Acceptance criteria (Phase 2)
- Example notebooks open and run end-to-end on CPU with provided data; badges work.
- `CITATION.cff` is present; the site has a “Cite” page and links to DOI for tagged releases.

### Phase 3 — Versioned documentation and releases (post v0.x)

1) Versioning workflow
  - Adopt `mike` for versioned MkDocs deployments (keep `latest` + `stable` + per-release versions).
  - Add a version switcher in the theme configuration.
  - Update release workflow to publish docs for tags/branches through `mike`.

2) Backfill and housekeeping
  - Publish at least the current release and `latest` development docs.
  - Document the policy for which versions are maintained.

Acceptance criteria (Phase 3)
- Docs site shows a version selector; visiting a tagged version shows frozen content.
- Release CI pushes versioned docs automatically on tags.

### Phase 4 — Quality, automation, and content depth (ongoing)

1) CI hardening
  - Add link checking (e.g., a fast link checker) and spell/style linters (markdownlint/codespell) in a separate job.
  - Fail on missing anchors or unresolved xrefs; keep strict builds.

2) API documentation quality
  - Introduce Doxygen groups/namespaces and brief descriptions for public headers.
  - Enable Graphviz diagrams in Doxygen; ensure headers compile for docs (standalone includes).
  - Add a “doc debt” label to track missing or outdated API comments.

3) CLI reference
  - Create `docs/reference/cli.md` listing all flags and defaults with examples.
  - Optional automation: generate from `--help` output during release builds; otherwise, maintain manually.

4) Benchmarks and performance notes
  - Add small, deterministic benchmarks with instructions to reproduce (CPU-only baseline).
  - Document expected ranges to help users sanity-check runs.

5) Community and governance
  - Add maintainers/ownership table (site page) and issue triage guidance.
  - Document security/reporting process; link from Support.

### Ownership and tracking

- Assign page owners (API, Tutorials, Reference, Concepts) in a short table within `docs/dev/release.md` or the Support page.
- Create tracking issues for each phase, with checklists mirroring the acceptance criteria above.

### Risks and mitigations

- GPU-only examples can complicate execution in hosted environments; prefer CPU-only notebooks with small grids.
- Large API surfaces can generate noisy docs; focus on public headers first and hide internal/private symbols.
- Versioned docs add CI complexity; keep `latest` + one stable version initially.
