"""Emit llms.txt and llms-full.txt from a generated documentation tree.

llms.txt is a machine-readable index of the documentation site
(https://llmstxt.org); llms-full.txt inlines every page so an agent can
load the whole site in one request. Both are built from the pages on
disk, so they stay correct after incremental updates.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union


def _page_title(page: Path) -> str:
    """First markdown heading of the page, or a name derived from its path."""
    try:
        for line in page.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    except OSError:
        pass
    return page.stem.replace("-", " ").replace("_", " ")


def _section(title: str, pages: List[Path], docs_root: Path) -> List[str]:
    if not pages:
        return []
    lines = ["", f"## {title}", ""]
    lines.extend(
        f"- [{_page_title(page)}]({page.relative_to(docs_root).as_posix()})"
        for page in pages
    )
    return lines


def write_llms_txt(
    output_dir: Union[str, Path], site_name: str, site_description: str
) -> Tuple[Path, Path]:
    """Write llms.txt and llms-full.txt into the docs tree.

    Returns the paths of the two written files.
    """
    docs_root = Path(output_dir) / "docs"
    docs_root.mkdir(parents=True, exist_ok=True)

    overview = docs_root / "index.md"
    guides = sorted((docs_root / "guides").glob("*.md"))
    modules = sorted((docs_root / "modules").rglob("index.md"))
    api_pages = sorted(
        page for page in (docs_root / "api").rglob("*.md") if page.name != "index.md"
    )
    api_index = docs_root / "api" / "index.md"

    lines = [f"# {site_name}", ""]
    if site_description:
        lines.extend([f"> {site_description}", ""])
    if overview.exists():
        lines.extend(["## Overview", "", f"- [{_page_title(overview)}](index.md)"])
    lines.extend(_section("Guides", guides, docs_root))
    lines.extend(_section("Modules", modules, docs_root))
    if api_index.exists():
        lines.extend(_section("API Reference", [api_index], docs_root))
    lines.extend(_section("Files", api_pages, docs_root))
    lines.append("")

    llms_path = docs_root / "llms.txt"
    llms_path.write_text("\n".join(lines), encoding="utf-8")

    all_pages = [page for page in (overview, *guides, *modules) if page.exists()]
    if api_index.exists():
        all_pages.append(api_index)
    all_pages.extend(api_pages)

    full_parts = [f"# {site_name}", ""]
    if site_description:
        full_parts.extend([f"> {site_description}", ""])
    for page in all_pages:
        rel = page.relative_to(docs_root).as_posix()
        full_parts.extend(
            ["", "---", "", f"<!-- {rel} -->", "", page.read_text(encoding="utf-8")]
        )
    full_path = docs_root / "llms-full.txt"
    full_path.write_text("\n".join(full_parts), encoding="utf-8")

    return llms_path, full_path
