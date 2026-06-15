"""Coverage for the FTS5-backed lexical store."""

from pathlib import Path

import pytest

from docstra.core.ingestion.fts_storage import FtsStorage, _sanitize_fts_query


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
    # Shape contract: metadata sub-dict must be present and document_id must match file_id.
    assert hits[0]["metadata"]["document_id"] == hits[0]["file_id"]


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


def test_add_and_search_symbols(tmp_path: Path):
    from docstra.core.indexing.model import IndexedSymbol

    store = FtsStorage(str(tmp_path / "index.db"))
    store.add_symbols([
        IndexedSymbol(
            id="repo/file.py::function::make_chunk_id::L67",
            file_id="repo/file.py",
            name="make_chunk_id",
            kind="function",
            language="python",
            line=67,
        ),
        IndexedSymbol(
            id="repo/other.py::class::CoreIndexBuilder::L194",
            file_id="repo/other.py",
            name="CoreIndexBuilder",
            kind="class",
            language="python",
            line=194,
        ),
    ])
    hits = store.search_symbols("CoreIndexBuilder", n_results=5)
    assert len(hits) == 1
    assert hits[0]["name"] == "CoreIndexBuilder"
    assert hits[0]["file_id"] == "repo/other.py"
    assert hits[0]["id"] == hits[0]["symbol_id"]
    assert hits[0]["metadata"]["document_id"] == hits[0]["file_id"]
    assert hits[0]["metadata"]["name"] == "CoreIndexBuilder"


def test_delete_by_file_removes_symbols(tmp_path: Path):
    from docstra.core.indexing.model import IndexedSymbol

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
    store.delete_by_file("x.py")
    assert store.search_symbols("foo", n_results=5) == []


def test_sanitize_fts_query_strips_special_chars():
    assert _sanitize_fts_query("What is the config?") == "what is the config"
    assert _sanitize_fts_query("(foo OR bar)") == "foo or bar"
    assert _sanitize_fts_query('search "exact phrase"') == "search exact phrase"
    assert _sanitize_fts_query("a*b") == "a b"
    assert _sanitize_fts_query("") == ""
    assert _sanitize_fts_query("???") == ""


def test_punctuation_query_does_not_raise(tmp_path: Path):
    store = FtsStorage(str(tmp_path / "index.db"))
    _add_chunk(store)
    # FTS5 would raise sqlite3.OperationalError without sanitization.
    hits = store.search_chunks("What is the config?", n_results=5)
    assert isinstance(hits, list)


def test_empty_query_returns_empty_list(tmp_path: Path):
    store = FtsStorage(str(tmp_path / "index.db"))
    _add_chunk(store)
    assert store.search_chunks("???", n_results=5) == []
    assert store.search_symbols("???", n_results=5) == []


def test_get_chunk_returns_row(tmp_path: Path):
    store = FtsStorage(str(tmp_path / "index.db"))
    payload = _add_chunk(store)
    row = store.get_chunk(payload["chunk_id"])
    assert row is not None
    assert row["chunk_id"] == payload["chunk_id"]
    assert row["content"] == payload["content"]


def test_get_chunk_returns_none_for_missing(tmp_path: Path):
    store = FtsStorage(str(tmp_path / "index.db"))
    assert store.get_chunk("nonexistent#L1-L1") is None


def test_query_with_reserved_words_does_not_raise(tmp_path: Path):
    """A natural-language query starting with NOT/AND/OR must not crash FTS5."""
    store = FtsStorage(str(tmp_path / "index.db"))
    _add_chunk(store, content="def handle_not_null(value): pass")
    # Each of these would otherwise be parsed as a FTS5 boolean expression.
    for query in ("NOT null handling", "OR operator", "AND something", "foo AND"):
        # Must not raise; result can be empty or have hits — we only care about no crash.
        store.search_chunks(query, n_results=5)
