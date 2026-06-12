"""Verify ingestion writes chunks and symbols into the FTS store alongside Chroma."""

from pathlib import Path
from typing import List

from docstra.core.document_processing.document import (
    CodeChunk,
    Document,
    DocumentMetadata,
    DocumentType,
)
from docstra.core.ingestion.fts_storage import FtsStorage
from docstra.core.ingestion.storage import ChromaDBStorage, DocumentIndexer


class _FakeEmbedder:
    """Returns a deterministic small vector so Chroma is happy without a real model."""

    def generate_embedding(self, _text: str) -> List[float]:
        return [0.0, 1.0, 0.0]


def _make_document(filepath: str, chunk_content: str) -> Document:
    metadata = DocumentMetadata(
        filepath=filepath,
        language=DocumentType.PYTHON,
        size_bytes=len(chunk_content.encode("utf-8")),
        last_modified=0.0,
        line_count=chunk_content.count("\n") + 1,
    )
    return Document(
        content=chunk_content,
        metadata=metadata,
        chunks=[
            CodeChunk(
                content=chunk_content,
                start_line=1,
                end_line=metadata.line_count,
                chunk_type="function",
            )
        ],
    )


def test_document_indexer_writes_chunks_to_fts(tmp_path: Path) -> None:
    chroma = ChromaDBStorage(persist_directory=str(tmp_path / "chroma"))
    fts = FtsStorage(str(tmp_path / "index.db"))

    document = _make_document("repo/foo.py", "def find_me(): pass")

    indexer = DocumentIndexer(
        chroma,
        embedding_generator=_FakeEmbedder(),
        codebase_root=str(tmp_path),
        fts_storage=fts,
    )
    indexer.index_document(document)

    hits = fts.search_chunks("find_me", n_results=5)
    assert len(hits) == 1
    assert hits[0]["file_id"] == "repo/foo.py"


def test_document_indexer_reindex_replaces_chunks(tmp_path: Path) -> None:
    """Re-indexing the same file should not duplicate chunks in the FTS store."""
    chroma = ChromaDBStorage(persist_directory=str(tmp_path / "chroma"))
    fts = FtsStorage(str(tmp_path / "index.db"))
    indexer = DocumentIndexer(
        chroma,
        embedding_generator=_FakeEmbedder(),
        codebase_root=str(tmp_path),
        fts_storage=fts,
    )

    indexer.index_document(_make_document("repo/foo.py", "def find_me(): pass"))
    indexer.index_document(_make_document("repo/foo.py", "def find_me(): pass"))

    hits = fts.search_chunks("find_me", n_results=5)
    assert len(hits) == 1
