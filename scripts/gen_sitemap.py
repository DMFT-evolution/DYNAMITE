#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urljoin


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a basic sitemap.xml from a MkDocs site/ directory.")
    ap.add_argument("--site-dir", default="site", help="Path to built site directory (default: site)")
    ap.add_argument("--site-url", required=True, help="Canonical site URL, e.g. https://dmft-evolution.github.io/DYNAMITE/")
    args = ap.parse_args()

    site_dir = Path(args.site_dir)
    site_url = args.site_url.strip()
    if not site_url.endswith("/"):
        site_url += "/"

    html_files = sorted({p.relative_to(site_dir).as_posix() for p in site_dir.rglob("*.html")})

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for rel in html_files:
        lines.append("  <url>")
        lines.append(f"    <loc>{urljoin(site_url, rel)}</loc>")
        lines.append("  </url>")
    lines.append("</urlset>")

    (site_dir / "sitemap.xml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
