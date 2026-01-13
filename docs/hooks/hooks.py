from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urljoin


def _iter_pages(files) -> list[str]:
    # MkDocs gives us File objects; use their dest_uri (relative path in site/).
    urls: list[str] = []
    for f in files:
        try:
            dest = getattr(f, "dest_uri", None)
            if not dest:
                continue
            # Skip non-HTML and common MkDocs artifacts
            if not dest.endswith(".html"):
                continue
            urls.append(dest)
        except Exception:
            continue
    return sorted(set(urls))


def on_post_build(config, **kwargs):
    """Write sitemap.xml and ensure robots.txt is present in site_dir.

    We do this without extra dependencies so the GitHub Pages workflow stays simple.
    """

    site_dir = Path(config["site_dir"])  # e.g. site/
    docs_dir = Path(config["docs_dir"])  # e.g. docs/

    site_url = (config.get("site_url") or "").strip()
    if site_url and not site_url.endswith("/"):
        site_url += "/"

    # robots.txt: copy from docs/ if present
    src_robots = docs_dir / "robots.txt"
    if src_robots.exists():
        (site_dir / "robots.txt").write_text(src_robots.read_text(encoding="utf-8"), encoding="utf-8")

    # sitemap.xml: walk built HTML files (robust even if nav excludes some pages)
    html_files = []
    for p in site_dir.rglob("*.html"):
        rel = p.relative_to(site_dir).as_posix()
        # Skip 404 and search index pages if desired; keep it simple and include all.
        html_files.append(rel)

    html_files = sorted(set(html_files))

    def loc(rel_path: str) -> str:
        if site_url:
            return urljoin(site_url, rel_path)
        # Fallback: relative URL (still useful for some tools)
        return rel_path

    # Minimal XML sitemap
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for rel in html_files:
        xml_lines.append("  <url>")
        xml_lines.append(f"    <loc>{loc(rel)}</loc>")
        xml_lines.append("  </url>")
    xml_lines.append("</urlset>")

    (site_dir / "sitemap.xml").write_text("\n".join(xml_lines) + "\n", encoding="utf-8")
