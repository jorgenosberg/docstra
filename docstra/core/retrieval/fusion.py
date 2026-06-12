"""Reciprocal Rank Fusion over dense + lexical retrieval sources."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Protocol

from docstra.core.indexing.code_index import CodebaseIndex


def rrf_score(rank: int, k: int) -> float:
    """Standard RRF contribution for a single source at 1-based rank."""
    return 1.0 / (k + rank)


class _DenseLike(Protocol):
    def retrieve_chunks(self, query: str, n_results: int = 20, **filters) -> List[Dict[str, Any]]: ...


class _FtsLike(Protocol):
    def retrieve_chunks(self, query: str, n_results: int = 50, **filters) -> List[Dict[str, Any]]: ...
    def retrieve_symbols(self, query: str, n_results: int = 25) -> List[Dict[str, Any]]: ...


class FusionRetriever:
    """Runs dense + lexical (chunks, symbols) retrieval and fuses with RRF."""

    def __init__(
        self,
        dense: _DenseLike,
        fts: _FtsLike,
        code_index: CodebaseIndex,
        *,
        rrf_k: int = 60,
        fts_chunks_top_k: int = 50,
        fts_symbols_top_k: int = 25,
    ) -> None:
        self.dense = dense
        self.fts = fts
        self.code_index = code_index
        self.rrf_k = rrf_k
        self.fts_chunks_top_k = fts_chunks_top_k
        self.fts_symbols_top_k = fts_symbols_top_k

    def retrieve(self, query: str, n_results: int = 20, **filters) -> List[Dict[str, Any]]:
        return self.retrieve_chunks(query, n_results=n_results, **filters)

    def retrieve_chunks(
        self, query: str, n_results: int = 20, **filters
    ) -> List[Dict[str, Any]]:
        dense_hits = self.dense.retrieve_chunks(query, n_results=n_results * 2, **filters)
        lex_chunk_hits = self.fts.retrieve_chunks(
            query, n_results=self.fts_chunks_top_k, **filters
        )
        symbol_hits = self.fts.retrieve_symbols(query, n_results=self.fts_symbols_top_k)
        symbol_chunk_hits = self._symbols_to_chunks(symbol_hits, filters)

        scored: Dict[str, float] = defaultdict(float)
        record: Dict[str, Dict[str, Any]] = {}
        for source in (dense_hits, lex_chunk_hits, symbol_chunk_hits):
            for rank, hit in enumerate(source, start=1):
                chunk_id = self._chunk_id(hit)
                if chunk_id is None:
                    continue
                scored[chunk_id] += rrf_score(rank, self.rrf_k)
                record.setdefault(chunk_id, hit)

        ordered = sorted(record.items(), key=lambda kv: (-scored[kv[0]], kv[0]))
        return [hit for _, hit in ordered[:n_results]]

    def retrieve_by_language(
        self, query: str, language: str, n_results: int = 20
    ) -> List[Dict[str, Any]]:
        return self.retrieve_chunks(query, n_results=n_results, language=language)

    def retrieve_code_examples(
        self, query: str, n_results: int = 10, languages: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("retrieve_code_examples is moved over in Task 7")

    def _chunk_id(self, hit: Dict[str, Any]) -> Optional[str]:
        return hit.get("chunk_id") or hit.get("id")

    def _symbols_to_chunks(
        self, symbol_hits: Iterable[Dict[str, Any]], filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        language = filters.get("language")
        file_id_filter = filters.get("file_id")
        results: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for symbol_hit in symbol_hits:
            file_id = symbol_hit.get("file_id")
            if file_id is None:
                continue
            if file_id_filter is not None and file_id != file_id_filter:
                continue
            if language is not None and self.code_index.file_language(file_id) != language:
                continue
            symbol_id = symbol_hit.get("symbol_id", "")
            line = _extract_line_from_symbol_id(symbol_id)
            if line is None:
                continue
            for chunk_id, start_line, end_line in self.code_index.chunks_for_file(file_id):
                if start_line <= line <= end_line and chunk_id not in seen:
                    seen.add(chunk_id)
                    results.append({
                        "chunk_id": chunk_id,
                        "id": chunk_id,
                        "file_id": file_id,
                        "start_line": start_line,
                        "end_line": end_line,
                        "content": "",
                        "metadata": {
                            "document_id": file_id,
                            "start_line": start_line,
                            "end_line": end_line,
                            "via_symbol": symbol_hit.get("name"),
                        },
                    })
                    break
        return results


def _extract_line_from_symbol_id(symbol_id: str) -> Optional[int]:
    if not symbol_id:
        return None
    tail = symbol_id.rsplit("::", 1)[-1]
    if not tail.startswith("L"):
        return None
    try:
        return int(tail[1:])
    except ValueError:
        return None
