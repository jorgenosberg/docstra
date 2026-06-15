"""Coverage for the FTS-backed retriever."""

from pathlib import Path

from docstra.core.indexing.model import IndexedSymbol
from docstra.core.ingestion.fts_storage import FtsStorage
from docstra.core.retrieval.fts import FtsRetriever


def test_retrieve_chunks_delegates_to_storage(tmp_path: Path):
    store = FtsStorage(str(tmp_path / "index.db"))
    store.add_chunks(
        chunk_ids=["repo/file.py#L1-L10"],
        file_ids=["repo/file.py"],
        languages=["python"],
        start_lines=[1],
        end_lines=[10],
        contents=["def make_chunk_id(): pass"],
    )
    retriever = FtsRetriever(store)
    hits = retriever.retrieve_chunks("make_chunk_id", n_results=5)
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "repo/file.py#L1-L10"
    assert hits[0]["metadata"]["document_id"] == hits[0]["file_id"]


def test_retrieve_symbols_delegates_to_storage(tmp_path: Path):
    store = FtsStorage(str(tmp_path / "index.db"))
    store.add_symbols([
        IndexedSymbol(
            id="x.py::function::foo::L1",
            file_id="x.py",
            name="foo",
            kind="function",
            language="python",
            line=1,
        )
    ])
    retriever = FtsRetriever(store)
    hits = retriever.retrieve_symbols("foo", n_results=5)
    assert len(hits) == 1
    assert hits[0]["name"] == "foo"
    assert hits[0]["metadata"]["document_id"] == "x.py"
