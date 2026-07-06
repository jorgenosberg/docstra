"""Cross-linking stage: render deterministic cross-reference sections."""

from __future__ import annotations

import posixpath
from typing import Dict, List, Mapping, Optional


def _render_entry(
    file_id: str,
    source_doc_path: Optional[str],
    target_doc_paths: Optional[Mapping[str, str]],
) -> str:
    """Render one cross-reference entry, as a relative link when possible."""
    if source_doc_path and target_doc_paths and file_id in target_doc_paths:
        href = posixpath.relpath(
            target_doc_paths[file_id], start=posixpath.dirname(source_doc_path)
        )
        return f"- [`{file_id}`]({href})"
    return f"- `{file_id}`"


def render_cross_references_section(
    cross_refs: Dict[str, List[str]],
    source_doc_path: Optional[str] = None,
    target_doc_paths: Optional[Mapping[str, str]] = None,
) -> str:
    """Render a cross-references section from graph data.

    Every entry comes from resolved import edges, so the section cannot
    reference files that do not exist. When ``source_doc_path`` and
    ``target_doc_paths`` are given, entries whose target has a generated page
    become relative markdown links; other entries stay plain code spans.
    """
    imports = sorted(cross_refs.get("imports", []))
    imported_by = sorted(cross_refs.get("imported_by", []))
    if not imports and not imported_by:
        return ""

    lines = ["", "", "## Cross-references", ""]
    if imports:
        lines.append("**Imports:**")
        lines.extend(
            _render_entry(file_id, source_doc_path, target_doc_paths)
            for file_id in imports
        )
    if imported_by:
        if imports:
            lines.append("")
        lines.append("**Imported by:**")
        lines.extend(
            _render_entry(file_id, source_doc_path, target_doc_paths)
            for file_id in imported_by
        )
    lines.append("")
    return "\n".join(lines)
