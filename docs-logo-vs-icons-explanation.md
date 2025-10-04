# Why Only the Logo Worked - Technical Explanation

## Key Insight

The logo works because of **how MkDocs handles theme assets** vs. **how browsers handle paths in markdown-generated HTML**.

## Theme Assets (Logo) - Automatic Handling

```yaml
# mkdocs.yml
theme:
  logo: assets/logo-dmfe.svg  # ← Works automatically
```

**What MkDocs does:**
1. Reads `assets/logo-dmfe.svg` from `docs/` directory
2. Copies it to `site/assets/logo-dmfe.svg`
3. Generates HTML with proper base-path-aware URLs
4. Result: Logo loads correctly on all pages

## Markdown Images (Icons) - Manual Path Required

```markdown
# Before (BROKEN):
<img src="../assets/icons/tag.svg" />

# After (FIXED):
<img src="/DMFE/assets/icons/tag.svg" />
```

**Why relative paths failed:**
- MkDocs converts markdown to HTML but **doesn't modify `<img src>` paths**
- Browser resolves relative paths from the **current page's URL**
- With `site_url: .../DMFE/`, pages are nested under `/DMFE/`
- Relative path `../assets/...` goes to wrong location

**Example breakdown:**

Page URL: `https://dmft-evolution.github.io/DMFE/concepts/version-compatibility/`

```
❌ Relative: src="../assets/icons/tag.svg"
   Resolves to: /DMFE/concepts/assets/icons/tag.svg (WRONG!)

✓ Absolute: src="/DMFE/assets/icons/tag.svg"  
   Resolves to: /DMFE/assets/icons/tag.svg (CORRECT!)
```

## The Fix

Use **absolute paths from site root** that match the `site_url` base path:

```markdown
/DMFE/assets/icons/filename.svg
 ^^^^^
 Must match site_url base path
```

This works because:
1. Browser treats `/DMFE/...` as absolute path from domain root
2. Path is the same regardless of page depth
3. Matches the actual asset location in deployed site

## Summary

| Asset Type | Config Location | Path Type | Reason |
|------------|----------------|-----------|--------|
| **Logo** (theme) | `mkdocs.yml` → `theme.logo` | Relative OK | MkDocs handles it |
| **Icons** (markdown) | `*.md` → `<img src>` | Must be absolute | Browser resolves it |

**Solution:** All markdown images must use absolute paths: `/DMFE/assets/icons/...`
