"""End-to-end tests: full generation, linked cross-references, incremental update."""

from __future__ import annotations

from pathlib import Path
from typing import List

from docstra.core.document_processing.document import (
    Document,
    DocumentMetadata,
    DocumentType,
)
from docstra.core.documentation.checks import check_documentation
from docstra.core.documentation.generator import DocumentationGenerator
from docstra.core.indexing.code_index import CodebaseIndex
from docstra.core.indexing.model import CoreIndexBuilder


class StubLlmClient:
    def __init__(self) -> None:
        self.response = "# Doc v1"

    def document_code(self, code: str, language: str, additional_context: str) -> str:
        del code, language, additional_context
        return self.response


def _make_document(repo: Path, name: str, content: str, imports: List[str]) -> Document:
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
            imports=imports,
            classes=[],
            functions=["f"],
            symbols={"f": [1]},
        ),
        chunks=[],
    )


def _setup(tmp_path: Path) -> tuple[List[Document], CodebaseIndex, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    documents = [
        _make_document(
            repo,
            "app.py",
            "import helpers\n\n\ndef f() -> None:\n    helpers.f()\n",
            imports=["import helpers"],
        ),
        _make_document(repo, "helpers.py", "def f() -> None:\n    pass\n", imports=[]),
        _make_document(
            repo, "standalone.py", "def f() -> None:\n    pass\n", imports=[]
        ),
    ]
    manifest = CoreIndexBuilder.from_documents(documents, codebase_root=repo)
    index = CodebaseIndex(
        index_directory=str(tmp_path / "index"), codebase_root=str(repo)
    )
    index.replace_manifest(manifest)
    return documents, index, tmp_path / "docs_out"


def test_generation_produces_linked_cross_references_that_pass_checks(
    tmp_path: Path,
) -> None:
    documents, index, output_dir = _setup(tmp_path)
    generator = DocumentationGenerator(
        llm_client=StubLlmClient(), output_dir=output_dir, code_index=index
    )

    assert generator.generate_documentation(
        documents, generate_guides=False, generate_api_docs=False
    )

    app_page = (output_dir / "docs" / "api" / "app.py.md").read_text(encoding="utf-8")
    assert "- [`helpers.py`](helpers.py.md)" in app_page

    helpers_page = (output_dir / "docs" / "api" / "helpers.py.md").read_text(
        encoding="utf-8"
    )
    assert "- [`app.py`](app.py.md)" in helpers_page

    report = check_documentation(output_dir, index)
    assert report.passed, [issue.to_dict() for issue in report.issues]
    assert report.cross_references_checked >= 2


def test_update_regenerates_only_impacted_pages(tmp_path: Path) -> None:
    documents, index, output_dir = _setup(tmp_path)
    stub = StubLlmClient()
    generator = DocumentationGenerator(
        llm_client=stub, output_dir=output_dir, code_index=index
    )
    assert generator.generate_documentation(
        documents, generate_guides=False, generate_api_docs=False
    )

    api_dir = output_dir / "docs" / "api"
    assert "# Doc v1" in (api_dir / "standalone.py.md").read_text(encoding="utf-8")

    # helpers.py changed: its page and its importer's page regenerate,
    # standalone.py's page must not be touched.
    stub.response = "# Doc v2"
    fresh_generator = DocumentationGenerator(
        llm_client=stub, output_dir=output_dir, code_index=index
    )
    assert fresh_generator.update_documentation(documents, {"helpers.py", "app.py"})

    assert "# Doc v2" in (api_dir / "helpers.py.md").read_text(encoding="utf-8")
    assert "# Doc v2" in (api_dir / "app.py.md").read_text(encoding="utf-8")
    assert "# Doc v1" in (api_dir / "standalone.py.md").read_text(encoding="utf-8")

    # Cross-reference links survive the update and still resolve.
    report = check_documentation(output_dir, index)
    assert report.passed, [issue.to_dict() for issue in report.issues]


def test_update_with_no_impacted_files_is_a_no_op(tmp_path: Path) -> None:
    documents, index, output_dir = _setup(tmp_path)
    stub = StubLlmClient()
    generator = DocumentationGenerator(
        llm_client=stub, output_dir=output_dir, code_index=index
    )
    assert generator.generate_documentation(
        documents, generate_guides=False, generate_api_docs=False
    )

    stub.response = "# Doc v2"
    assert generator.update_documentation(documents, set())

    api_dir = output_dir / "docs" / "api"
    for name in ("app.py.md", "helpers.py.md", "standalone.py.md"):
        assert "# Doc v1" in (api_dir / name).read_text(encoding="utf-8")
