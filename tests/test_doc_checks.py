"""Tests for the deterministic documentation checks."""

from __future__ import annotations

from pathlib import Path
from typing import List

from docstra.core.document_processing.document import (
    Document,
    DocumentMetadata,
    DocumentType,
)
from docstra.core.documentation.checks import check_documentation
from docstra.core.indexing.code_index import CodebaseIndex
from docstra.core.indexing.model import CoreIndexBuilder


def _make_index(tmp_path: Path, file_names: List[str]) -> CodebaseIndex:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    documents = []
    for name in file_names:
        path = repo / name
        content = "def f() -> None:\n    pass\n"
        path.write_text(content)
        documents.append(
            Document(
                content=content,
                metadata=DocumentMetadata(
                    filepath=str(path),
                    language=DocumentType.PYTHON,
                    size_bytes=len(content),
                    last_modified=1.0,
                    line_count=2,
                    imports=[],
                    classes=[],
                    functions=["f"],
                    symbols={},
                ),
                chunks=[],
            )
        )
    manifest = CoreIndexBuilder.from_documents(documents, codebase_root=repo)
    index = CodebaseIndex(
        index_directory=str(tmp_path / "index"), codebase_root=str(repo)
    )
    index.replace_manifest(manifest)
    return index


def test_clean_docs_pass_all_checks(tmp_path: Path) -> None:
    index = _make_index(tmp_path, ["a.py", "b.py"])

    output_dir = tmp_path / "out"
    api_dir = output_dir / "docs" / "api"
    api_dir.mkdir(parents=True)
    (output_dir / "docs" / "index.md").write_text(
        "# Home\n\nSee [a](api/a.py.md) and [external](https://example.com).\n"
    )
    (api_dir / "a.py.md").write_text(
        "# a.py\n\n\n## Cross-references\n\n**Imports:**\n- [`b.py`](b.py.md)\n"
    )
    (api_dir / "b.py.md").write_text(
        "# b.py\n\n\n## Cross-references\n\n**Imported by:**\n- `a.py`\n"
    )

    report = check_documentation(output_dir, index)

    assert report.passed
    assert report.pages_checked == 3
    assert report.links_checked == 2
    assert report.cross_references_checked == 2


def test_dead_links_and_unresolved_cross_references_fail(tmp_path: Path) -> None:
    index = _make_index(tmp_path, ["a.py"])

    output_dir = tmp_path / "out"
    docs_dir = output_dir / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "index.md").write_text(
        "# Home\n\n[missing](missing.md)\n\n"
        "## Cross-references\n\n**Imports:**\n- `ghost.py`\n"
    )

    report = check_documentation(output_dir, index)

    assert not report.passed
    checks = {issue.check for issue in report.issues}
    assert checks == {"dead-link", "unresolved-cross-reference"}
    details = " ".join(issue.detail for issue in report.issues)
    assert "missing.md" in details
    assert "ghost.py" in details


def test_anchor_and_external_links_are_ignored(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    docs_dir = output_dir / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "index.md").write_text(
        "# Home\n\n[anchor](#section)\n[mail](mailto:a@b.c)\n[web](https://x.y)\n"
    )

    report = check_documentation(output_dir)

    assert report.passed
    assert report.links_checked == 0
