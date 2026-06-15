from __future__ import annotations

from pathlib import Path

import pytest

from docstra.core.document_processing.document import (
    CodeChunk,
    Document,
    DocumentMetadata,
    DocumentType,
)
from docstra.core.indexing.code_index import CodebaseIndex, CodebaseIndexer
from docstra.core.indexing.model import (
    CORE_INDEX_FILENAME,
    CoreIndexBuilder,
    CoreIndexManifest,
    IndexedChunk,
    IndexedFile,
    IndexedSymbol,
    ImportRecord,
)
from docstra.core.ingestion.storage import ChromaDBStorage, DocumentIndexer


class DummyEmbeddingGenerator:
    def generate_embedding(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0, 0.0]

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def _make_document(
    path: Path,
    *,
    content: str,
    imports: list[str],
    functions: list[str],
    classes: list[str],
    symbols: dict[str, list[int]],
    chunks: list[CodeChunk],
) -> Document:
    return Document(
        content=content,
        metadata=DocumentMetadata(
            filepath=str(path),
            language=DocumentType.PYTHON,
            size_bytes=len(content.encode("utf-8")),
            last_modified=1.0,
            line_count=len(content.splitlines()),
            imports=imports,
            classes=classes,
            functions=functions,
            symbols=symbols,
        ),
        chunks=chunks,
    )


def test_core_index_manifest_round_trip() -> None:
    manifest = CoreIndexManifest(
        embedding_backend="chroma",
        embedding_model="test-model",
        source_kinds=["tree-sitter"],
        files=[
            IndexedFile(
                id="docstra/core/cli.py",
                language="python",
                size_bytes=100,
                last_modified=1.0,
                line_count=10,
            )
        ],
        chunks=[
            IndexedChunk(
                id="docstra/core/cli.py#L1-L4",
                file_id="docstra/core/cli.py",
                language="python",
                start_line=1,
                end_line=4,
                chunk_type="function",
                symbols=["main"],
            )
        ],
        symbols=[
            IndexedSymbol(
                id="docstra/core/cli.py::function::main::L1",
                file_id="docstra/core/cli.py",
                name="main",
                kind="function",
                language="python",
                line=1,
            )
        ],
        imports=[
            ImportRecord(
                id="docstra/core/cli.py::import::0",
                source_file_id="docstra/core/cli.py",
                raw_text="from docstra.core.app import app",
                target_file_id="docstra/core/app.py",
            )
        ],
    )

    payload = manifest.model_dump_json(indent=2)
    restored = CoreIndexManifest.model_validate_json(payload)

    assert restored.embedding_model == "test-model"
    assert restored.files[0].id == "docstra/core/cli.py"
    assert restored.chunks[0].id == "docstra/core/cli.py#L1-L4"
    assert restored.symbols[0].id == "docstra/core/cli.py::function::main::L1"
    assert restored.imports[0].target_file_id == "docstra/core/app.py"


def test_core_index_builder_creates_stable_ids_and_edges(tmp_path: Path) -> None:
    codebase_root = tmp_path / "repo"
    codebase_root.mkdir()
    helper_dir = codebase_root / "pkg"
    helper_dir.mkdir()

    consumer_path = codebase_root / "consumer.py"
    helper_path = helper_dir / "helper.py"
    consumer_content = "def run():\n    return util()\n"
    helper_content = "def util():\n    return 1\n"
    consumer_path.write_text(consumer_content, encoding="utf-8")
    helper_path.write_text(helper_content, encoding="utf-8")

    consumer = _make_document(
        consumer_path,
        content=consumer_content,
        imports=["from pkg.helper import util"],
        functions=["run"],
        classes=[],
        symbols={"run": [1]},
        chunks=[
            CodeChunk(
                content=consumer_content,
                start_line=1,
                end_line=2,
                symbols=["run"],
                chunk_type="function",
                parent_symbols=[],
            )
        ],
    )
    helper = _make_document(
        helper_path,
        content=helper_content,
        imports=[],
        functions=["util"],
        classes=[],
        symbols={"util": [1]},
        chunks=[
            CodeChunk(
                content=helper_content,
                start_line=1,
                end_line=2,
                symbols=["util"],
                chunk_type="function",
                parent_symbols=[],
            )
        ],
    )

    manifest = CoreIndexBuilder.from_documents(
        [consumer, helper],
        codebase_root,
        embedding_backend="chroma",
        embedding_model="test-embed",
    )

    assert sorted(indexed_file.id for indexed_file in manifest.files) == [
        "consumer.py",
        "pkg/helper.py",
    ]
    assert {chunk.id for chunk in manifest.chunks} == {
        "consumer.py#L1-L2",
        "pkg/helper.py#L1-L2",
    }
    assert {symbol.id for symbol in manifest.symbols} == {
        "consumer.py::function::run::L1",
        "pkg/helper.py::function::util::L1",
    }
    assert manifest.imports[0].target_file_id == "pkg/helper.py"
    assert manifest.edges[0].source_id == "consumer.py"
    assert manifest.edges[0].target_id == "pkg/helper.py"
    assert {embedding.vector_id for embedding in manifest.embeddings} >= {
        "consumer.py",
        "consumer.py#L1-L2",
        "pkg/helper.py",
        "pkg/helper.py#L1-L2",
    }

    index_dir = codebase_root / ".docstra" / "index"
    indexer = CodebaseIndexer(
        index_directory=str(index_dir),
        codebase_root=str(codebase_root),
        embedding_model="test-embed",
    )
    indexer.index_documents([consumer, helper])
    code_index = CodebaseIndex(
        index_directory=str(index_dir), codebase_root=str(codebase_root)
    )

    assert (index_dir / CORE_INDEX_FILENAME).exists()
    assert code_index.get_file_dependencies("consumer.py") == ["pkg/helper.py"]
    assert code_index.get_related_files("pkg/helper.py") == ["consumer.py"]
    assert code_index.search_function("util")[0]["filepath"] == "pkg/helper.py"
    assert code_index.get_file_metadata(str(helper_path))["filepath"] == "pkg/helper.py"


def test_codebase_indexer_index_document_preserves_existing_files(
    tmp_path: Path,
) -> None:
    codebase_root = tmp_path / "repo"
    codebase_root.mkdir()
    helper_dir = codebase_root / "pkg"
    helper_dir.mkdir()

    helper_path = helper_dir / "helper.py"
    consumer_path = codebase_root / "consumer.py"

    helper_content = "def util():\n    return 1\n"
    helper_updated_content = "def util():\n    value = 1\n    return value\n"
    consumer_content = "from pkg.helper import util\n\ndef run():\n    return util()\n"

    helper_path.write_text(helper_updated_content, encoding="utf-8")
    consumer_path.write_text(consumer_content, encoding="utf-8")

    helper = _make_document(
        helper_path,
        content=helper_content,
        imports=[],
        functions=["util"],
        classes=[],
        symbols={"util": [1]},
        chunks=[
            CodeChunk(
                content=helper_content,
                start_line=1,
                end_line=2,
                symbols=["util"],
                chunk_type="function",
                parent_symbols=[],
            )
        ],
    )
    helper_updated = _make_document(
        helper_path,
        content=helper_updated_content,
        imports=[],
        functions=["util"],
        classes=[],
        symbols={"util": [1]},
        chunks=[
            CodeChunk(
                content=helper_updated_content,
                start_line=1,
                end_line=3,
                symbols=["util"],
                chunk_type="function",
                parent_symbols=[],
            )
        ],
    )
    consumer = _make_document(
        consumer_path,
        content=consumer_content,
        imports=["from pkg.helper import util"],
        functions=["run"],
        classes=[],
        symbols={"run": [3]},
        chunks=[
            CodeChunk(
                content=consumer_content,
                start_line=3,
                end_line=4,
                symbols=["run"],
                chunk_type="function",
                parent_symbols=[],
            )
        ],
    )

    index_dir = codebase_root / ".docstra" / "index"
    indexer = CodebaseIndexer(
        index_directory=str(index_dir),
        codebase_root=str(codebase_root),
        embedding_model="test-embed",
    )

    indexer.index_document(helper)
    indexer.index_document(consumer)
    indexer.index_document(helper_updated)

    code_index = CodebaseIndex(
        index_directory=str(index_dir), codebase_root=str(codebase_root)
    )

    assert sorted(code_index.iter_file_ids()) == ["consumer.py", "pkg/helper.py"]
    assert code_index.get_file_dependencies("consumer.py") == ["pkg/helper.py"]
    assert code_index.get_related_files("pkg/helper.py") == ["consumer.py"]
    assert code_index.search_function("util")[0]["filepath"] == "pkg/helper.py"
    assert code_index.get_file_metadata("pkg/helper.py")["line_count"] == 3


def test_core_index_builder_resolves_all_python_multi_import_targets(
    tmp_path: Path,
) -> None:
    codebase_root = tmp_path / "repo"
    codebase_root.mkdir()
    pkg_dir = codebase_root / "pkg"
    pkg_dir.mkdir()

    init_path = pkg_dir / "__init__.py"
    module_a_path = pkg_dir / "a.py"
    module_b_path = pkg_dir / "b.py"
    consumer_path = codebase_root / "consumer.py"

    init_content = "from .a import alpha\nfrom .b import beta\n"
    module_a_content = "def alpha():\n    return 'a'\n"
    module_b_content = "def beta():\n    return 'b'\n"
    consumer_content = "from pkg import a, b\nimport pkg.a, pkg.b\n"

    for path, content in [
        (init_path, init_content),
        (module_a_path, module_a_content),
        (module_b_path, module_b_content),
        (consumer_path, consumer_content),
    ]:
        path.write_text(content, encoding="utf-8")

    package_init = _make_document(
        init_path,
        content=init_content,
        imports=["from .a import alpha", "from .b import beta"],
        functions=[],
        classes=[],
        symbols={},
        chunks=[],
    )
    module_a = _make_document(
        module_a_path,
        content=module_a_content,
        imports=[],
        functions=["alpha"],
        classes=[],
        symbols={"alpha": [1]},
        chunks=[
            CodeChunk(
                content=module_a_content,
                start_line=1,
                end_line=2,
                symbols=["alpha"],
                chunk_type="function",
                parent_symbols=[],
            )
        ],
    )
    module_b = _make_document(
        module_b_path,
        content=module_b_content,
        imports=[],
        functions=["beta"],
        classes=[],
        symbols={"beta": [1]},
        chunks=[
            CodeChunk(
                content=module_b_content,
                start_line=1,
                end_line=2,
                symbols=["beta"],
                chunk_type="function",
                parent_symbols=[],
            )
        ],
    )
    consumer = _make_document(
        consumer_path,
        content=consumer_content,
        imports=["from pkg import a, b", "import pkg.a, pkg.b"],
        functions=[],
        classes=[],
        symbols={},
        chunks=[],
    )

    manifest = CoreIndexBuilder.from_documents(
        [package_init, module_a, module_b, consumer],
        codebase_root,
    )

    from_import_records = [
        record
        for record in manifest.imports
        if record.raw_text == "from pkg import a, b"
    ]
    import_records = [
        record
        for record in manifest.imports
        if record.raw_text == "import pkg.a, pkg.b"
    ]

    assert {record.target_file_id for record in from_import_records} == {
        "pkg/a.py",
        "pkg/b.py",
    }
    assert {record.target_file_id for record in import_records} == {
        "pkg/a.py",
        "pkg/b.py",
    }
    assert {
        (edge.source_id, edge.target_id)
        for edge in manifest.edges
        if edge.source_id == "consumer.py"
    } == {
        ("consumer.py", "pkg/a.py"),
        ("consumer.py", "pkg/b.py"),
    }
    assert "pkg/__init__.py" not in {
        record.target_file_id for record in from_import_records
    }


def test_document_indexer_stores_repo_relative_document_ids(tmp_path: Path) -> None:
    codebase_root = tmp_path / "repo"
    codebase_root.mkdir()
    source_path = codebase_root / "app.py"
    source_content = "def main():\n    return 1\n"
    source_path.write_text(source_content, encoding="utf-8")

    document = _make_document(
        source_path,
        content=source_content,
        imports=[],
        functions=["main"],
        classes=[],
        symbols={"main": [1]},
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

    storage = ChromaDBStorage(persist_directory=str(tmp_path / "chroma"))
    indexer = DocumentIndexer(
        storage,
        DummyEmbeddingGenerator(),
        codebase_root=str(codebase_root),
    )

    doc_id = indexer.index_document(document)
    doc_record = storage.get_document("app.py")
    chunk_records = storage.get_chunks_for_document("app.py")

    assert doc_id == "app.py"
    assert doc_record is not None
    assert doc_record["id"] == "app.py"
    assert doc_record["metadata"]["document_id"] == "app.py"
    assert doc_record["metadata"]["filepath"] == "app.py"
    assert chunk_records[0]["id"] == "app.py#L1-L2"
    assert chunk_records[0]["metadata"]["document_id"] == "app.py"


def test_codebase_index_rejects_legacy_sidecars_without_core_manifest(
    tmp_path: Path,
) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "file_index.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Rerun 'docstra ingest'"):
        CodebaseIndex(index_directory=str(index_dir), codebase_root=str(tmp_path))


def test_chunks_for_file_returns_chunks_in_line_order(tmp_path):
    from docstra.core.indexing.code_index import CodebaseIndex
    from docstra.core.indexing.model import CoreIndexManifest, IndexedChunk

    index = CodebaseIndex(index_directory=str(tmp_path / "index"))
    index._manifest = CoreIndexManifest.empty()
    index._manifest.chunks.extend([
        IndexedChunk(id="a.py#L1-L10", file_id="a.py", language="python",
                     start_line=1, end_line=10, chunk_type="code"),
        IndexedChunk(id="a.py#L11-L20", file_id="a.py", language="python",
                     start_line=11, end_line=20, chunk_type="code"),
        IndexedChunk(id="b.py#L1-L5", file_id="b.py", language="python",
                     start_line=1, end_line=5, chunk_type="code"),
    ])
    index._rebuild_lookups()

    assert index.chunks_for_file("a.py") == [("a.py#L1-L10", 1, 10), ("a.py#L11-L20", 11, 20)]
    assert index.chunks_for_file("missing.py") == []

    # Verify sorting is enforced even when chunks are inserted in reverse line order.
    index2 = CodebaseIndex(index_directory=str(tmp_path / "index2"))
    index2._manifest = CoreIndexManifest.empty()
    index2._manifest.chunks.extend([
        IndexedChunk(id="c.py#L100-L110", file_id="c.py", language="python",
                     start_line=100, end_line=110, chunk_type="code"),
        IndexedChunk(id="c.py#L1-L10", file_id="c.py", language="python",
                     start_line=1, end_line=10, chunk_type="code"),
    ])
    index2._rebuild_lookups()

    result = index2.chunks_for_file("c.py")
    assert result == [("c.py#L1-L10", 1, 10), ("c.py#L100-L110", 100, 110)]
