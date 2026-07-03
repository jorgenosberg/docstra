"""Tests for graph-based cross-references in the documentation generator."""

from __future__ import annotations

from pathlib import Path

from docstra.core.document_processing.document import (
    Document,
    DocumentMetadata,
    DocumentType,
)
from docstra.core.documentation.generator import DocumentationGenerator
from docstra.core.indexing.code_index import CodebaseIndex
from docstra.core.indexing.model import CoreIndexBuilder


class StubLlmClient:
    def document_code(self, code: str, language: str, additional_context: str) -> str:
        del code, language
        self.last_context = additional_context
        return "# Generated documentation"


def _make_document(repo: Path, name: str, content: str) -> Document:
    filepath = repo / name
    filepath.write_text(content)
    return Document(
        content=content,
        metadata=DocumentMetadata(
            filepath=str(filepath),
            language=DocumentType.PYTHON,
            size_bytes=len(content.encode("utf-8")),
            last_modified=1.0,
            line_count=len(content.splitlines()),
            imports=["import helpers"] if name == "app.py" else [],
            classes=[],
            functions=["main"] if name == "app.py" else ["helper"],
            symbols={"main": [4]} if name == "app.py" else {"helper": [1]},
        ),
        chunks=[],
    )


def _make_index(tmp_path: Path, repo: Path, documents: list[Document]) -> CodebaseIndex:
    manifest = CoreIndexBuilder.from_documents(documents, codebase_root=repo)
    index = CodebaseIndex(
        index_directory=str(tmp_path / "index"), codebase_root=str(repo)
    )
    index.replace_manifest(manifest)
    return index


def _setup(tmp_path: Path) -> tuple[DocumentationGenerator, Document, Document]:
    repo = tmp_path / "repo"
    repo.mkdir()
    helpers_doc = _make_document(
        repo, "helpers.py", "def helper() -> int:\n    return 1\n"
    )
    app_doc = _make_document(
        repo,
        "app.py",
        "import helpers\n\n\ndef main() -> int:\n    return helpers.helper()\n",
    )
    index = _make_index(tmp_path, repo, [app_doc, helpers_doc])
    generator = DocumentationGenerator(
        llm_client=StubLlmClient(),
        output_dir=tmp_path / "docs_out",
        code_index=index,
    )
    return generator, app_doc, helpers_doc


def test_cross_references_come_from_import_graph(tmp_path: Path) -> None:
    generator, app_doc, helpers_doc = _setup(tmp_path)

    app_refs = generator._get_file_cross_references(app_doc)
    assert app_refs["imports"] == ["helpers.py"]

    helpers_refs = generator._get_file_cross_references(helpers_doc)
    assert helpers_refs["imported_by"] == ["app.py"]


def test_cross_references_context_labels_directions(tmp_path: Path) -> None:
    generator, app_doc, helpers_doc = _setup(tmp_path)

    app_context = generator._get_cross_references_context(app_doc)
    assert "helpers.py" in app_context
    assert "Imports" in app_context

    helpers_context = generator._get_cross_references_context(helpers_doc)
    assert "app.py" in helpers_context
    assert "Imported By" in helpers_context


def test_render_cross_references_section() -> None:
    section = DocumentationGenerator._render_cross_references_section(
        {"imports": ["helpers.py"], "imported_by": ["app.py"]}
    )
    assert "## Cross-references" in section
    assert "- `helpers.py`" in section
    assert "- `app.py`" in section

    assert DocumentationGenerator._render_cross_references_section({}) == ""
    assert (
        DocumentationGenerator._render_cross_references_section(
            {"imports": [], "imported_by": []}
        )
        == ""
    )


def test_file_doc_gets_deterministic_cross_reference_section(tmp_path: Path) -> None:
    generator, app_doc, _ = _setup(tmp_path)

    generator._generate_single_file_doc(app_doc)

    written = list((tmp_path / "docs_out" / "docs" / "api").rglob("app.py.md"))
    assert len(written) == 1
    content = written[0].read_text(encoding="utf-8")
    assert content.startswith("# Generated documentation")
    assert "## Cross-references" in content
    assert "- `helpers.py`" in content


def test_missing_code_index_yields_empty_cross_references(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    doc = _make_document(repo, "app.py", "def main() -> None:\n    pass\n")
    generator = DocumentationGenerator(
        llm_client=StubLlmClient(), output_dir=tmp_path / "docs_out"
    )

    assert generator._get_file_cross_references(doc) == {}
    assert generator._get_cross_references_context(doc) == "None identified"
