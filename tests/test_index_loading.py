from __future__ import annotations

from pathlib import Path

from docstra.core.config.settings import UserConfig
from docstra.core.document_processing.document import (
    CodeChunk,
    Document,
    DocumentMetadata,
    DocumentType,
)
from docstra.core.indexing.model import CORE_INDEX_FILENAME, CoreIndexBuilder
from docstra.core.ingestion.storage import ChromaDBStorage
from docstra.core.services.query_service import QueryService
from docstra.core.services.repository_explorer_service import RepositoryExplorerService


class DummyEmbeddingGenerator:
    def generate_embedding(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0, 0.0]

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def _write_core_index(codebase_root: Path) -> None:
    source_path = codebase_root / "app.py"
    source_content = "def main():\n    return 1\n"
    source_path.write_text(source_content, encoding="utf-8")

    document = Document(
        content=source_content,
        metadata=DocumentMetadata(
            filepath=str(source_path),
            language=DocumentType.PYTHON,
            size_bytes=len(source_content.encode("utf-8")),
            last_modified=1.0,
            line_count=2,
            imports=[],
            classes=[],
            functions=["main"],
            symbols={"main": [1]},
        ),
        chunks=[
            CodeChunk(
                content=source_content,
                start_line=1,
                end_line=2,
                symbols=["main"],
                chunk_type="function",
                parent_symbols=[],
            )
        ],
    )

    persist_dir = codebase_root / ".docstra"
    index_dir = persist_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest = CoreIndexBuilder.from_documents([document], codebase_root)
    (index_dir / CORE_INDEX_FILENAME).write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    ChromaDBStorage(persist_directory=str(persist_dir / "chroma"))


def test_query_service_initializes_from_core_index_without_repo_map(
    tmp_path: Path, monkeypatch
) -> None:
    codebase_root = tmp_path / "repo"
    codebase_root.mkdir()
    _write_core_index(codebase_root)

    monkeypatch.setattr(
        "docstra.core.services.query_service._get_llm_client_for_service",
        lambda config, callbacks=None: object(),
    )
    monkeypatch.setattr(
        "docstra.core.services.query_service.EmbeddingFactory.create_embedding_generator",
        lambda embedding_type, **kwargs: DummyEmbeddingGenerator(),
    )

    config = UserConfig()
    config.storage.persist_directory = ".docstra"
    service = QueryService(config)
    service._ensure_retrieval_components_initialized(codebase_root.resolve())

    assert service.code_indexer is not None
    assert service.context_aware_retriever is not None
    assert service.context_aware_retriever.repo_map is not None
    assert not (codebase_root / ".docstra" / "repo_map.json").exists()


def test_repository_explorer_service_loads_core_index_without_repo_map(
    tmp_path: Path,
) -> None:
    codebase_root = tmp_path / "repo"
    codebase_root.mkdir()
    _write_core_index(codebase_root)

    config = UserConfig()
    config.storage.persist_directory = ".docstra"
    service = RepositoryExplorerService(config)
    service._load_components(str(codebase_root))

    assert service.code_index is not None
    assert service.repo_map is not None
    assert service.code_index.get_file_metadata(str(codebase_root / "app.py")) == {
        "filepath": "app.py",
        "language": "python",
        "size_bytes": len("def main():\n    return 1\n".encode("utf-8")),
        "line_count": 2,
        "last_modified": 1.0,
        "classes": [],
        "functions": ["main"],
        "imports": [],
        "module_docstring": None,
        "dependencies": [],
        "dependents": [],
        "complexity": 1,
        "complexity_metrics": {},
        "code_quality": {},
        "documentation_coverage": None,
        "test_coverage": None,
        "category": None,
        "contributors": [],
        "tags": [],
    }
    assert service.repo_map.find_file("app.py") is not None
