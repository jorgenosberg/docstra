from __future__ import annotations

from docstra.core.ingestion.storage import ChromaDBStorage


def test_chroma_storage_round_trip_with_upgraded_client(tmp_path) -> None:
    storage = ChromaDBStorage(persist_directory=str(tmp_path / "chroma"))

    storage.add_document(
        document_id="docstra/core/cli.py",
        content="print('hello')",
        metadata={"document_id": "docstra/core/cli.py", "kind": "document"},
        embedding=[1.0, 0.0, 0.0],
    )
    storage.add_chunks(
        chunk_ids=["docstra/core/cli.py#1-1"],
        contents=["print('hello')"],
        metadatas=[
            {
                "document_id": "docstra/core/cli.py",
                "kind": "chunk",
                "symbols": ["print"],
            }
        ],
        embeddings=[[1.0, 0.0, 0.0]],
    )

    documents = storage.search_documents([1.0, 0.0, 0.0], n_results=1)
    chunks = storage.search_chunks([1.0, 0.0, 0.0], n_results=1)

    assert documents[0]["id"] == "docstra/core/cli.py"
    assert documents[0]["metadata"]["kind"] == "document"
    assert chunks[0]["id"] == "docstra/core/cli.py#1-1"
    assert chunks[0]["metadata"]["document_id"] == "docstra/core/cli.py"
