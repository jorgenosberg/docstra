"""Deterministic quality checks for generated documentation.

These checks require no LLM: they verify that internal markdown links
resolve to real files and that cross-reference entries name files that
exist in the core index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from docstra.core.indexing.code_index import CodebaseIndex

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
CROSS_REF_ENTRY_RE = re.compile(r"^- \[?`([^`]+)`")


@dataclass(frozen=True)
class DocsCheckIssue:
    """One failed check on one documentation page."""

    check: str  # "dead-link" | "unresolved-cross-reference"
    doc_path: str  # page path relative to the docs root
    detail: str

    def to_dict(self) -> Dict[str, str]:
        return {"check": self.check, "doc_path": self.doc_path, "detail": self.detail}


@dataclass
class DocsCheckReport:
    """Aggregate result of all documentation checks."""

    issues: List[DocsCheckIssue] = field(default_factory=list)
    pages_checked: int = 0
    links_checked: int = 0
    cross_references_checked: int = 0

    @property
    def passed(self) -> bool:
        return not self.issues

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "pages_checked": self.pages_checked,
            "links_checked": self.links_checked,
            "cross_references_checked": self.cross_references_checked,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _is_external_link(target: str) -> bool:
    return "://" in target or target.startswith(("mailto:", "tel:"))


def _iter_internal_links(content: str) -> List[str]:
    """Return link targets that point at other files in the docs tree."""
    links = []
    for match in MARKDOWN_LINK_RE.finditer(content):
        target = match.group(1)
        if _is_external_link(target) or target.startswith("#"):
            continue
        links.append(target)
    return links


def _cross_reference_entries(content: str) -> List[str]:
    """Return the file ids listed in the page's cross-references section."""
    entries = []
    in_section = False
    for line in content.splitlines():
        if line.strip() == "## Cross-references":
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section:
            match = CROSS_REF_ENTRY_RE.match(line.strip())
            if match:
                entries.append(match.group(1))
    return entries


def check_documentation(
    output_dir: Path, code_index: Optional[CodebaseIndex] = None
) -> DocsCheckReport:
    """Run deterministic checks over a generated documentation tree.

    Checks every markdown page for dead internal links, and, when a code
    index is available, verifies that every cross-reference entry names a
    file that exists in the index.
    """
    docs_root = output_dir / "docs"
    if not docs_root.exists():
        docs_root = output_dir

    known_file_ids = set(code_index.iter_file_ids()) if code_index else None

    report = DocsCheckReport()
    for page in sorted(docs_root.rglob("*.md")):
        report.pages_checked += 1
        rel_page = str(page.relative_to(docs_root))
        content = page.read_text(encoding="utf-8")

        for target in _iter_internal_links(content):
            report.links_checked += 1
            path_part = target.split("#", 1)[0]
            if not path_part:
                continue
            if path_part.startswith("/"):
                resolved = docs_root / path_part.lstrip("/")
            else:
                resolved = page.parent / path_part
            if not resolved.exists():
                report.issues.append(
                    DocsCheckIssue(
                        check="dead-link",
                        doc_path=rel_page,
                        detail=f"link target does not exist: {target}",
                    )
                )

        if known_file_ids is not None:
            for file_id in _cross_reference_entries(content):
                report.cross_references_checked += 1
                if file_id not in known_file_ids:
                    report.issues.append(
                        DocsCheckIssue(
                            check="unresolved-cross-reference",
                            doc_path=rel_page,
                            detail=f"cross-reference not in the code index: {file_id}",
                        )
                    )

    return report
