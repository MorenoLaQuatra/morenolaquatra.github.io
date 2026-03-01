#!/usr/bin/env python3
"""
Auto-generate sitemap.xml by scanning HTML files in the site directory.

Usage:
    python generate_sitemap.py

Scans all .html files, excludes MARP slides and non-site pages,
assigns priorities based on directory depth, and writes sitemap.xml.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent

# ── Configuration ──────────────────────────────────────────────

SITE_URL = "https://mlaquatra.me"
SITE_DIR = Path(__file__).parent

# Files/patterns to exclude from sitemap
EXCLUDE_PATTERNS = [
    "galactica_llm_slides.html",  # MARP slides
    "BDHPD-2.html",               # MARP slides
    "slides_preview.html",         # MARP slides
    "PACE.html",                   # standalone
    "languages.html",              # old LaTeX.css demo
    "src/",                        # old source files
    "node_modules/",
]

# Priority by path depth: root pages get higher priority
PRIORITY_MAP = {
    "index.html": "1.0",
    "cv.html": "0.9",
    "publications.html": "0.9",
    "teaching.html": "0.8",
    "projects.html": "0.8",
    "blog.html": "0.7",
    "talks.html": "0.7",
}

# Default priorities by directory
DIR_PRIORITY = {
    ".": "0.6",          # root-level misc pages
    "projects": "0.6",
    "posts": "0.6",
    "guides": "0.5",
}

# Change frequency by directory
DIR_CHANGEFREQ = {
    ".": "monthly",
    "projects": "yearly",
    "posts": "yearly",
    "guides": "yearly",
}


# ── Scanning ───────────────────────────────────────────────────

def should_exclude(rel_path: str) -> bool:
    for pattern in EXCLUDE_PATTERNS:
        if pattern in rel_path:
            return True
    return False


def has_component_system(filepath: Path) -> bool:
    """Check if the file uses the site's component system (data-include)."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        return 'data-include=' in content
    except Exception:
        return False


def scan_pages() -> list[dict]:
    pages = []

    for html_file in sorted(SITE_DIR.rglob("*.html")):
        rel = html_file.relative_to(SITE_DIR)
        rel_str = str(rel)

        if should_exclude(rel_str):
            continue

        # Only include pages that use the component system (part of the site)
        if not has_component_system(html_file):
            continue

        # Determine directory key
        parent = str(rel.parent)
        if parent == ".":
            dir_key = "."
        else:
            dir_key = parent.split("/")[0]

        filename = rel.name
        priority = PRIORITY_MAP.get(filename, DIR_PRIORITY.get(dir_key, "0.5"))
        changefreq = DIR_CHANGEFREQ.get(dir_key, "monthly")

        # News archives get lower priority
        if filename.startswith("news_"):
            priority = "0.4"
            changefreq = "yearly"

        # Build URL
        url_path = rel_str
        if url_path == "index.html":
            url_path = ""

        pages.append({
            "loc": f"{SITE_URL}/{url_path}",
            "priority": priority,
            "changefreq": changefreq,
        })

    return pages


# ── XML Generation ─────────────────────────────────────────────

def generate_sitemap(pages: list[dict]) -> None:
    urlset = Element("urlset")
    urlset.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")

    for page in pages:
        url_el = SubElement(urlset, "url")
        SubElement(url_el, "loc").text = page["loc"]
        SubElement(url_el, "changefreq").text = page["changefreq"]
        SubElement(url_el, "priority").text = page["priority"]

    indent(urlset, space="  ")

    tree = ElementTree(urlset)
    output = SITE_DIR / "sitemap.xml"
    tree.write(str(output), encoding="unicode", xml_declaration=True)

    # Add trailing newline
    with open(output, "a") as f:
        f.write("\n")

    print(f"Generated {output} with {len(pages)} URLs:")
    for page in pages:
        print(f"  [{page['priority']}] {page['loc']}")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    pages = scan_pages()
    generate_sitemap(pages)
