"""Coverage for RRF fusion of dense + lexical retrieval."""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from docstra.core.retrieval.fusion import FusionRetriever, rrf_score


class _FakeDense:
    def __init__(self, results: List[Dict[str, Any]]):
        self._results = results

    def retrieve_chunks(self, query: str, n_results: int = 20, **filters):
        return self._results[:n_results]


class _FakeFts:
    def __init__(self, chunks: List[Dict[str, Any]], symbols: List[Dict[str, Any]]):
        self._chunks = chunks
        self._symbols = symbols

    def retrieve_chunks(self, query: str, n_results: int = 50, **filters):
        return self._chunks[:n_results]

    def retrieve_symbols(self, query: str, n_results: int = 25):
        return self._symbols[:n_results]


def _chunk(chunk_id: str, file_id: str, start_line: int = 1, end_line: int = 10):
    return {
        "chunk_id": chunk_id,
        "id": chunk_id,
        "file_id": file_id,
        "language": "python",
        "start_line": start_line,
        "end_line": end_line,
        "content": f"# chunk {chunk_id}",
        "metadata": {"document_id": file_id, "start_line": start_line, "end_line": end_line},
    }


def _fake_code_index(chunks_by_file):
    def chunks_for_file(file_id):
        return chunks_by_file.get(file_id, [])

    return SimpleNamespace(chunks_for_file=chunks_for_file, file_language=lambda fid: None)


def test_rrf_score_known_inputs():
    assert rrf_score(rank=1, k=60) == pytest.approx(1 / 61)
    assert rrf_score(rank=5, k=60) == pytest.approx(1 / 65)


def test_fusion_orders_by_combined_rank():
    a = _chunk("repo/a.py#L1-L10", "repo/a.py")
    b = _chunk("repo/b.py#L1-L10", "repo/b.py")
    c = _chunk("repo/c.py#L1-L10", "repo/c.py")

    # dense: [a(rank1), b(rank2), c(rank3)]
    # fts chunks: [b(rank1), c(rank2), a(rank3)]
    # a: 1/61 + 1/63, b: 1/61 + 1/62, c: 1/63 + 1/62
    # b has the highest combined score, a and c tie below it
    dense = _FakeDense([a, b, c])
    fts = _FakeFts(chunks=[b, c, a], symbols=[])

    fusion = FusionRetriever(
        dense=dense,
        fts=fts,
        code_index=_fake_code_index({}),
        rrf_k=60,
        fts_chunks_top_k=10,
        fts_symbols_top_k=10,
    )
    hits = fusion.retrieve_chunks("anything", n_results=3)
    ids = [h["chunk_id"] for h in hits]
    assert ids[0] == "repo/b.py#L1-L10"
    assert set(ids[1:]) == {"repo/a.py#L1-L10", "repo/c.py#L1-L10"}


def test_symbol_hit_promotes_containing_chunk():
    a = _chunk("repo/a.py#L1-L10", "repo/a.py", start_line=1, end_line=10)
    b = _chunk("repo/b.py#L1-L10", "repo/b.py")

    dense = _FakeDense([b, a])
    fts = _FakeFts(
        chunks=[],
        symbols=[{"symbol_id": "repo/a.py::function::foo::L5", "file_id": "repo/a.py", "name": "foo", "kind": "function"}],
    )

    code_index = _fake_code_index({"repo/a.py": [("repo/a.py#L1-L10", 1, 10)]})

    fusion = FusionRetriever(
        dense=dense,
        fts=fts,
        code_index=code_index,
        rrf_k=60,
        fts_chunks_top_k=10,
        fts_symbols_top_k=10,
    )
    hits = fusion.retrieve_chunks("foo", n_results=2)
    assert hits[0]["chunk_id"] == "repo/a.py#L1-L10"


def test_symbol_path_respects_language_filter():
    """Symbol-derived chunks must obey the language filter just like dense/lex chunks."""
    py_chunk = _chunk("repo/a.py#L1-L10", "repo/a.py", start_line=1, end_line=10)
    ts_chunk = _chunk("repo/b.ts#L1-L10", "repo/b.ts", start_line=1, end_line=10)

    dense = _FakeDense([py_chunk])
    fts = _FakeFts(
        chunks=[],
        symbols=[
            {"symbol_id": "repo/a.py::function::foo::L5", "file_id": "repo/a.py", "name": "foo", "kind": "function"},
            {"symbol_id": "repo/b.ts::function::foo::L3", "file_id": "repo/b.ts", "name": "foo", "kind": "function"},
        ],
    )

    code_index = SimpleNamespace(
        chunks_for_file=lambda fid: {
            "repo/a.py": [("repo/a.py#L1-L10", 1, 10)],
            "repo/b.ts": [("repo/b.ts#L1-L10", 1, 10)],
        }.get(fid, []),
        file_language=lambda fid: {"repo/a.py": "python", "repo/b.ts": "typescript"}.get(fid),
    )

    fusion = FusionRetriever(
        dense=dense,
        fts=fts,
        code_index=code_index,
        rrf_k=60,
        fts_chunks_top_k=10,
        fts_symbols_top_k=10,
    )

    hits = fusion.retrieve_chunks("foo", n_results=5, language="python")
    ids = [h["chunk_id"] for h in hits]
    assert "repo/b.ts#L1-L10" not in ids
    assert "repo/a.py#L1-L10" in ids


def test_empty_lexical_source_does_not_break_fusion():
    a = _chunk("repo/a.py#L1-L10", "repo/a.py")
    dense = _FakeDense([a])
    fts = _FakeFts(chunks=[], symbols=[])
    fusion = FusionRetriever(
        dense=dense,
        fts=fts,
        code_index=_fake_code_index({}),
        rrf_k=60,
        fts_chunks_top_k=10,
        fts_symbols_top_k=10,
    )
    hits = fusion.retrieve_chunks("anything", n_results=5)
    assert [h["chunk_id"] for h in hits] == ["repo/a.py#L1-L10"]
