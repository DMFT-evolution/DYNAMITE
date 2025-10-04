# Icon Path Fix - Final Solution

## Root Cause

The logo (`logo-dmfe.svg`) works because it's referenced in `mkdocs.yml` theme configuration:
```yaml
theme:
  logo: assets/logo-dmfe.svg
```

MkDocs automatically handles theme asset paths. However, **icons in markdown `<img>` tags** are NOT processed the same way.

## The Problem with Relative Paths

When `site_url: https://dmft-evolution.github.io/DMFE/` includes a subpath (`/DMFE/`), relative paths in HTML don't resolve correctly:

### Example from `docs/concepts/version-compatibility.md`

**❌ BROKEN** (relative path):
```markdown
src="../assets/icons/tag.svg"
```

When built, the page is at: `/DMFE/concepts/version-compatibility/index.html`

The browser resolves `../assets/icons/tag.svg` as:
- Starting from: `/DMFE/concepts/version-compatibility/`
- Going up one level: `/DMFE/concepts/`
- Appending path: `/DMFE/concepts/assets/icons/tag.svg` ❌ (doesn't exist)

### What Actually Works

**✅ CORRECT** (absolute path from site root):
```markdown
src="/DMFE/assets/icons/tag.svg"
```

The browser interprets this as an absolute path from the domain root:
- `https://dmft-evolution.github.io/DMFE/assets/icons/tag.svg` ✓

## The Solution

All icon paths now use **absolute paths** starting with `/DMFE/`:

```markdown
# <img src="/DMFE/assets/icons/XYZ.svg" alt="Icon"/> Title
```

This matches the `site_url` base path and works correctly on:
- GitHub Pages deployment
- Local `mkdocs serve` 
- Any page depth in the site structure

## Files Updated

### Root-level pages (2 files)
- `docs/install.md`
- `docs/usage.md`

### Subfolder pages (20+ files)
- All files in `docs/concepts/` (7 files)
- All files in `docs/tutorials/` (5 files)
- All files in `docs/howto/` (4 files)
- All files in `docs/dev/` (3 files)
- `docs/reference/index.md`

### Documentation
- Updated `docs/howto/using-icons.md` with correct path pattern

## Verification

```bash
# Check all icons use absolute paths
grep -rh 'src="[^"]*icons/' docs/*.md docs/*/*.md | grep -c '/DMFE/assets/icons/'
# Should match the total number of icon usages

# Check no relative paths remain  
grep -r 'src="\.\./.*icons/' docs/ | wc -l
# Should output: 0
```

## Why This Works

1. **Absolute paths** (`/DMFE/...`) are resolved from the domain root
2. **Works on any page** regardless of nesting depth
3. **Matches site_url** configuration in `mkdocs.yml`
4. **Same pattern as working logo** (conceptually - the logo uses MkDocs theme handling, but the absolute path principle is the same)

## Testing

When you rebuild and deploy:
```bash
mkdocs build --strict
# No 404 warnings for icons

mkdocs serve
# Visit any page - all icons should load correctly
```

On GitHub Pages at `https://dmft-evolution.github.io/DMFE/`, all icons will now load correctly!
