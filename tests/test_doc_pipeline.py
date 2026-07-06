"""Tests for the documentation pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from docstra.core.document_processing.document import (
    Document,
    DocumentMetadata,
    DocumentType,
)
from docstra.core.documentation.pipeline import (
    analyze_codebase,
    compute_impacted_file_ids,
    plan_documentation,
    render_cross_references_section,
)
from docstra.core.indexing.code_index import CodebaseIndex
from docstra.core.indexing.model import CoreIndexBuilder


def _make_document(
    repo: Path,
    name: str,
    content: str,
    imports: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    functions: Optional[List[str]] = None,
) -> Document:
    filepath = repo / name
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content)
    return Document(
        content=content,
        metadata=DocumentMetadata(
            filepath=str(filepath),
            language=DocumentType.PYTHON,
            size_bytes=len(content.encode("utf-8")),
            last_modified=1.0,
            line_count=len(content.splitlines()),
            imports=imports or [],
            classes=classes or [],
            functions=functions or [],
            symbols={},
        ),
        chunks=[],
    )


def _make_index(tmp_path: Path, repo: Path, documents: List[Document]) -> CodebaseIndex:
    manifest = CoreIndexBuilder.from_documents(documents, codebase_root=repo)
    index = CodebaseIndex(
        index_directory=str(tmp_path / "index"), codebase_root=str(repo)
    )
    index.replace_manifest(manifest)
    return index


def test_analyze_groups_documents_by_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    docs = [
        _make_document(repo, "pkg/a.py", "def a() -> None:\n    pass\n"),
        _make_document(repo, "pkg/b.py", "def b() -> None:\n    pass\n"),
        _make_document(repo, "lib/c.py", "def c() -> None:\n    pass\n"),
    ]

    analysis = analyze_codebase(docs)

    assert set(analysis.module_structure) == {"pkg", "lib"}
    assert len(analysis.module_structure["pkg"]) == 2
    assert analysis.total_files == 3
    assert analysis.total_lines == 6


def test_plan_produces_expected_pages(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    docs = [
        _make_document(repo, "pkg/api.py", "class Api:\n    pass\n", classes=["Api"]),
        _make_document(repo, "pkg/util.py", "def util() -> None:\n    pass\n"),
    ]
    index = _make_index(tmp_path, repo, docs)
    analysis = analyze_codebase(docs)

    plan = plan_documentation(analysis, index)

    paths = {page.output_path for page in plan.pages}
    assert "docs/index.md" in paths
    assert "docs/modules/pkg/index.md" in paths
    assert "docs/api/pkg/api.py.md" in paths
    assert "docs/api/pkg/util.py.md" in paths
    assert "docs/guides/getting-started.md" in paths
    # api.py has a class, so an API index page is planned
    assert "docs/api/index.md" in paths

    file_paths = plan.file_page_paths()
    assert file_paths["pkg/api.py"] == "docs/api/pkg/api.py.md"


def test_plan_without_guides_or_api_index(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    docs = [_make_document(repo, "util.py", "def util() -> None:\n    pass\n")]
    analysis = analyze_codebase(docs)

    plan = plan_documentation(analysis, include_guides=False, include_api_docs=False)

    kinds = {page.kind for page in plan.pages}
    assert kinds == {"overview", "module", "file"}


def test_cross_references_render_as_relative_links() -> None:
    section = render_cross_references_section(
        {"imports": ["lib/b.py"], "imported_by": ["c.py"]},
        source_doc_path="docs/api/pkg/a.py.md",
        target_doc_paths={
            "lib/b.py": "docs/api/lib/b.py.md",
            "c.py": "docs/api/c.py.md",
        },
    )

    assert "- [`lib/b.py`](../lib/b.py.md)" in section
    assert "- [`c.py`](../c.py.md)" in section


def test_cross_references_fall_back_to_plain_entries() -> None:
    section = render_cross_references_section(
        {"imports": ["unknown.py"]},
        source_doc_path="docs/api/a.py.md",
        target_doc_paths={},
    )

    assert "- `unknown.py`" in section
    assert "](" not in section


def test_impact_includes_graph_neighbors_in_both_directions(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    docs = [
        _make_document(
            repo,
            "app.py",
            "import helpers\n\n\ndef main() -> None:\n    helpers.helper()\n",
            imports=["import helpers"],
        ),
        _make_document(repo, "helpers.py", "def helper() -> None:\n    pass\n"),
        _make_document(repo, "standalone.py", "def alone() -> None:\n    pass\n"),
    ]
    index = _make_index(tmp_path, repo, docs)

    impacted = compute_impacted_file_ids({"helpers.py"}, [index])
    assert impacted == {"helpers.py", "app.py"}

    impacted = compute_impacted_file_ids({"app.py"}, [index])
    assert impacted == {"app.py", "helpers.py"}

    impacted = compute_impacted_file_ids({"standalone.py"}, [index])
    assert impacted == {"standalone.py"}
