"""Coverage for the FTS5-backed lexical store."""

from pathlib import Path

import pytest

from docstra.core.ingestion.fts_storage import FtsStorage


def _add_chunk(store: FtsStorage, **overrides):
    payload = dict(
        chunk_id="repo/file.py#L1-L10",
        file_id="repo/file.py",
        language="python",
        start_line=1,
        end_line=10,
        content="def make_chunk_id(file_id, start_line, end_line):\n    return f'{file_id}#L{start_line}-L{end_line}'",
    )
    payload.update(overrides)
    store.add_chunks(
        chunk_ids=[payload["chunk_id"]],
        file_ids=[payload["file_id"]],
        languages=[payload["language"]],
        start_lines=[payload["start_line"]],
        end_lines=[payload["end_line"]],
        contents=[payload["content"]],
    )
    return payload


def test_schema_creates_on_first_open(tmp_path: Path):
    db_path = tmp_path / "index.db"
    store = FtsStorage(str(db_path))
    assert db_path.exists()
    # Idempotent: opening again does not raise.
    FtsStorage(str(db_path))


def test_add_and_search_chunks(tmp_path: Path):
    store = FtsStorage(str(tmp_path / "index.db"))
    _add_chunk(store)
    hits = store.search_chunks("make_chunk_id", n_results=5)
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "repo/file.py#L1-L10"
    assert hits[0]["file_id"] == "repo/file.py"
    assert hits[0]["start_line"] == 1
    assert hits[0]["end_line"] == 10
    assert "make_chunk_id" in hits[0]["content"]


def test_delete_by_file_removes_chunks_and_fts(tmp_path: Path):
    store = FtsStorage(str(tmp_path / "index.db"))
    _add_chunk(store)
    store.delete_by_file("repo/file.py")
    assert store.search_chunks("make_chunk_id", n_results=5) == []


def test_search_supports_language_filter(tmp_path: Path):
    store = FtsStorage(str(tmp_path / "index.db"))
    _add_chunk(store, chunk_id="a.py#L1-L1", file_id="a.py", language="python", content="foo bar")
    _add_chunk(store, chunk_id="b.ts#L1-L1", file_id="b.ts", language="typescript", content="foo bar")
    hits = store.search_chunks("foo", n_results=5, language="python")
    assert {h["file_id"] for h in hits} == {"a.py"}
