# Before & After Examples

## Example 1: version-compatibility.md

### BEFORE (Broken)
```markdown
# <img class="icon icon-lg icon-primary" src="../docs/assets/icons/tag.svg" alt="Version tag icon"/> Version compatibility

Policy:
- Identical: code_version in file equals current code_version → resume allowed silently.
- Compatible: major.minor match (first two numeric parts equal) → resume allowed; warnings may be emitted if git branch/hash differ.

Behavior and sources:
- Version info sources: `include/version/version_info.hpp` and `src/version/version_info.cpp`
```

**Issues**:
1. Icon path has `../docs/assets/icons/` → resolves to `/DMFE/docs/assets/icons/tag.svg` (404)
2. "Policy:" followed immediately by list → renders as plain text, not a bulleted list
3. "Behavior and sources:" same issue

### AFTER (Fixed)
```markdown
# <img class="icon icon-lg icon-primary" src="../assets/icons/tag.svg" alt="Version tag icon"/> Version compatibility

## Policy

- Identical: code_version in file equals current code_version → resume allowed silently.
- Compatible: major.minor match (first two numeric parts equal) → resume allowed; warnings may be emitted if git branch/hash differ.

## Behavior and sources

- Version info sources: `include/version/version_info.hpp` and `src/version/version_info.cpp`
```

**Fixes**:
1. Icon path now `../assets/icons/` → resolves to `/DMFE/assets/icons/tag.svg` ✓
2. "Policy" is now a proper `## Heading` → list renders correctly with bullets ✓
3. "Behavior and sources" is now a proper `## Heading` → list renders correctly ✓

---

## Example 2: time-integration.md

### BEFORE (Broken)
```markdown
Summary:
- RK54 = Dormand–Prince (adaptive default): embedded error estimate controls local error and proposes Δt.
- SSPRK(10,4) (auto‑switch): engaged when RK54 reaches its absolute‑stability bound

CLI controls:
- `-e` error tolerance (ε) for the embedded estimate.
- `-d` minimum time step to avoid over‑refinement near initial transients.
```

**Issue**: No blank line after "Summary:" and "CLI controls:" → lists don't render

### AFTER (Fixed)
```markdown
## Summary

- RK54 = Dormand–Prince (adaptive default): embedded error estimate controls local error and proposes Δt.
- SSPRK(10,4) (auto‑switch): engaged when RK54 reaches its absolute‑stability bound

## CLI controls

- `-e` error tolerance (ε) for the embedded estimate.
- `-d` minimum time step to avoid over‑refinement near initial transients.
```

**Fix**: Proper headings → lists render correctly with bullets ✓

---

## Why These Issues Occurred

### Icon Paths
The paths included `/docs/` which is not part of the site structure. When MkDocs builds:
- Source: `docs/concepts/version-compatibility.md`
- Built to: `site/concepts/version-compatibility/index.html`
- Assets: `docs/assets/` → `site/assets/`

So from `site/concepts/version-compatibility/index.html`:
- ❌ `../docs/assets/icons/tag.svg` resolves to `site/concepts/docs/assets/icons/tag.svg` (doesn't exist)
- ✓ `../assets/icons/tag.svg` resolves to `site/assets/icons/tag.svg` (correct!)

### List Rendering
Markdown requires either:
1. A blank line before a list, OR
2. The preceding line to be a heading

```markdown
❌ Text:
- Item 1

✓ Text:

- Item 1

✓ ## Heading
- Item 1
```

We used option #2 (convert to headings) for better document structure.
