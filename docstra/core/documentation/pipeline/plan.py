"""Planning stage: decide which documentation pages exist and where they live."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

from docstra.core.indexing.code_index import CodebaseIndex
from docstra.core.documentation.pipeline.analyze import CodebaseAnalysis

GUIDE_PAGES: List[Tuple[str, str]] = [
    ("getting-started", "Getting Started Guide"),
    ("installation", "Installation Instructions"),
    ("configuration", "Configuration Guide"),
    ("troubleshooting", "Troubleshooting Guide"),
]


@dataclass(frozen=True)
class PlannedPage:
    """One page of the documentation site."""

    kind: str  # "overview" | "module" | "file" | "guide" | "api_index"
    output_path: str  # posix path relative to the output directory
    title: str
    source_file_ids: Tuple[str, ...] = ()


@dataclass
class DocPlan:
    """The full set of pages to generate."""

    pages: List[PlannedPage] = field(default_factory=list)

    @property
    def file_pages(self) -> List[PlannedPage]:
        return [page for page in self.pages if page.kind == "file"]

    @property
    def module_pages(self) -> List[PlannedPage]:
        return [page for page in self.pages if page.kind == "module"]

    def file_page_paths(self) -> Dict[str, str]:
        """Map source file id to the output path of its file page."""
        return {
            page.source_file_ids[0]: page.output_path
            for page in self.file_pages
            if page.source_file_ids
        }


def doc_relative_path(filepath: str, code_index: Optional[CodebaseIndex] = None) -> str:
    """Map a source file path to a repo-relative doc path inside the output dir."""
    if code_index:
        normalized = code_index.normalize_file_id(filepath)
        if normalized and ".." not in Path(normalized).parts:
            return normalized

    rel_path = os.path.relpath(filepath, start=".")
    if ".." in Path(rel_path).parts:
        return Path(filepath).name
    return rel_path


def file_doc_path(filepath: str, code_index: Optional[CodebaseIndex] = None) -> str:
    """Return the output path of the file page for a source file."""
    rel = doc_relative_path(filepath, code_index)
    return str(PurePosixPath("docs") / "api" / f"{rel}.md")


def module_doc_path(module_name: str) -> str:
    """Return the output path of a module's index page."""
    slug = module_name.lower().replace(" ", "_")
    return str(PurePosixPath("docs") / "modules" / slug / "index.md")


def plan_documentation(
    analysis: CodebaseAnalysis,
    code_index: Optional[CodebaseIndex] = None,
    include_guides: bool = True,
    include_api_docs: bool = True,
) -> DocPlan:
    """Build the page plan for an analyzed codebase."""
    pages: List[PlannedPage] = [
        PlannedPage(
            kind="overview",
            output_path="docs/index.md",
            title="Overview",
            source_file_ids=tuple(
                doc_relative_path(doc.metadata.filepath, code_index)
                for doc in analysis.documents
            ),
        )
    ]

    for module_name, docs in analysis.module_structure.items():
        pages.append(
            PlannedPage(
                kind="module",
                output_path=module_doc_path(module_name),
                title=module_name.title(),
                source_file_ids=tuple(
                    doc_relative_path(doc.metadata.filepath, code_index) for doc in docs
                ),
            )
        )

    for doc in analysis.documents:
        file_id = doc_relative_path(doc.metadata.filepath, code_index)
        pages.append(
            PlannedPage(
                kind="file",
                output_path=file_doc_path(doc.metadata.filepath, code_index),
                title=Path(doc.metadata.filepath).name,
                source_file_ids=(file_id,),
            )
        )

    if include_guides:
        for guide_name, guide_title in GUIDE_PAGES:
            pages.append(
                PlannedPage(
                    kind="guide",
                    output_path=f"docs/guides/{guide_name}.md",
                    title=guide_title,
                )
            )

    if include_api_docs:
        api_files = tuple(
            doc_relative_path(doc.metadata.filepath, code_index)
            for doc in analysis.documents
            if len(doc.metadata.classes) > 0 or len(doc.metadata.functions) > 3
        )
        if api_files:
            pages.append(
                PlannedPage(
                    kind="api_index",
                    output_path="docs/api/index.md",
                    title="API Reference",
                    source_file_ids=api_files,
                )
            )

    return DocPlan(pages=pages)
