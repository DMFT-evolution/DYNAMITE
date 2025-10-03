#!/usr/bin/env bash
set -euo pipefail

# Generate Doxygen XML and convert to MkDocs-friendly Markdown via doxybook2.
# Requires: doxygen, doxybook2 (pip install doxybook2)

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)
cd "$ROOT_DIR"

# 1) Run doxygen
mkdir -p docs/reference/doxygen docs/reference/api
if ! command -v doxygen >/dev/null 2>&1; then
  echo "ERROR: doxygen not found. Install it (apt install doxygen)." >&2
  exit 1
fi

doxygen Doxyfile

# 2) Run doxybook2 if available; otherwise create a placeholder so MkDocs can build

# Resolve doxybook2 binary (prefer DOXYBOOK2 env var, else PATH)
DOXYBOOK2_BIN="${DOXYBOOK2:-$(command -v doxybook2 || true)}"

if [ -n "$DOXYBOOK2_BIN" ] && [ -x "$DOXYBOOK2_BIN" ]; then
  rm -rf docs/reference/api/*

  # Resolve templates directory: support explicit DOXYBOOK2_TEMPLATES or discover next to the binary
  TEMPLATES_OPT=""
  if [ -n "${DOXYBOOK2_TEMPLATES:-}" ] && [ -d "$DOXYBOOK2_TEMPLATES" ]; then
    TEMPLATES_OPT=(--templates "$DOXYBOOK2_TEMPLATES")
  else
    DOXY_DIR=$(dirname "$DOXYBOOK2_BIN")
    for cand in \
      "$DOXY_DIR/templates/default" \
      "$DOXY_DIR/../templates/default" \
      "docs/reference/templates/default"; do
      if [ -d "$cand" ]; then
        TEMPLATES_OPT=(--templates "$cand")
        break
      fi
    done
  fi

  echo "Using doxybook2 at: $DOXYBOOK2_BIN"
  if [ -n "${TEMPLATES_OPT[*]:-}" ]; then
    echo "Using templates at: ${TEMPLATES_OPT[1]}"
  else
    echo "No templates directory found; proceeding without --templates (if supported by this doxybook2)."
  fi

  # shellcheck disable=SC2068
  "$DOXYBOOK2_BIN" --input docs/reference/doxygen/xml \
    --output docs/reference/api \
    --config docs/reference/doxybook2.json \
    ${TEMPLATES_OPT[@]:-}

  # 3) Post-process MkDocs links to avoid Dir/Dir duplication
  # - Self links: (Dir/file.md#anchor) -> (#anchor)
  # - Same-dir links: (Dir/other.md#anchor) -> (other.md#anchor)
  while IFS= read -r -d '' mdfile; do
    dir_name=$(basename "$(dirname "$mdfile")")
    base_name=$(basename "$mdfile")
    # Replace (Dir/file.md#...) with (#...)
    sed -i "s#(${dir_name}/${base_name}#(#g" "$mdfile"
    # Replace (Dir/xxx.md#...) with (xxx.md#...)
    sed -i "s#(${dir_name}/#(#g" "$mdfile"
  done < <(find docs/reference/api -type f -name "*.md" -print0)

  echo "API docs generated under docs/reference/api/"
else
  echo "WARN: doxybook2 not found. Skipping API conversion and writing placeholder." >&2
  mkdir -p docs/reference/api
  cat > docs/reference/api/index.md <<'MD'
# API Reference (Placeholder)

Doxygen XML was generated to `docs/reference/doxygen/xml`.

To render Markdown API pages, install doxybook2 and re-run:

```bash
./docs/gen_api.sh
```

You can also set DOXYBOOK2 to the full path of the binary, and DOXYBOOK2_TEMPLATES to its templates/default directory.
MD
fi
