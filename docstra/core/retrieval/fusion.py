"""Reciprocal Rank Fusion over dense + lexical retrieval sources."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Protocol

from docstra.core.indexing.code_index import CodebaseIndex


def rrf_score(rank: int, k: int) -> float:
    """Standard RRF contribution for a single source at 1-based rank."""
    return 1.0 / (k + rank)


class _DenseLike(Protocol):
    def retrieve_chunks(
        self, query: str, n_results: int = 20, **filters
    ) -> List[Dict[str, Any]]: ...


class _FtsLike(Protocol):
    def retrieve_chunks(
        self, query: str, n_results: int = 50, **filters
    ) -> List[Dict[str, Any]]: ...
    def retrieve_symbols(
        self, query: str, n_results: int = 25
    ) -> List[Dict[str, Any]]: ...
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]: ...


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

    def retrieve(
        self, query: str, n_results: int = 20, **filters
    ) -> List[Dict[str, Any]]:
        return self.retrieve_chunks(query, n_results=n_results, **filters)

    def retrieve_chunks(
        self, query: str, n_results: int = 20, **filters
    ) -> List[Dict[str, Any]]:
        dense_hits = self.dense.retrieve_chunks(
            query, n_results=n_results * 2, **filters
        )
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
        """Retrieve chunks that are good examples of the queried concept.

        Args:
            query: Query string
            n_results: Number of results to return
            languages: Optional list of languages to filter by

        Returns:
            List of example chunks
        """
        # Start with basic vector search
        filters = {}
        if languages:
            # We'll retrieve for each language separately
            all_results = []
            for language in languages:
                results = self.retrieve_by_language(
                    query=query,
                    language=language,
                    n_results=max(n_results // len(languages), 1),
                )
                all_results.extend(results)

            vector_results = all_results
        else:
            vector_results = self.retrieve_chunks(
                query=query,
                n_results=n_results * 2,  # Get more for filtering
                **filters,
            )

        # Filter for chunks that are likely to be good examples
        # - Prefer complete functions/methods
        # - Prefer moderately sized chunks (not too short, not too long)
        # - Prefer chunks with meaningful names
        good_examples = []

        for chunk in vector_results:
            chunk_type = chunk["metadata"].get("chunk_type", "")
            content = chunk["content"]

            # Score the chunk as an example
            example_score = 0.0

            # Prefer functions/methods
            if chunk_type in ["function", "method"]:
                example_score += 1.0

            # Check content length (not too short, not too long)
            lines = content.count("\n") + 1
            if 5 <= lines <= 50:
                example_score += 0.5

            # Look for meaningful names (more than 3 characters, not generic)
            symbols = chunk["metadata"].get("symbols", [])
            generic_symbols = [
                "main",
                "init",
                "test",
                "get",
                "set",
                "run",
                "func",
                "foo",
                "bar",
            ]

            for symbol in symbols:
                if len(symbol) > 3 and symbol.lower() not in generic_symbols:
                    example_score += 0.3
                    break

            chunk_id = self._chunk_id(chunk)
            if example_score > 0:
                good_examples.append(
                    {
                        "chunk_id": chunk_id,
                        "id": chunk_id,
                        "content": content,
                        "metadata": chunk["metadata"],
                        "score": example_score,
                    }
                )

        # Sort by combined score and return top results
        sorted_examples = sorted(
            good_examples,
            key=lambda x: x.get("score", 0),
            reverse=True,  # Higher score is better
        )

        return sorted_examples[:n_results]

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
            if (
                language is not None
                and self.code_index.file_language(file_id) != language
            ):
                continue
            symbol_id = symbol_hit.get("symbol_id", "")
            line = _extract_line_from_symbol_id(symbol_id)
            if line is None:
                continue
            for chunk_id, start_line, end_line in self.code_index.chunks_for_file(
                file_id
            ):
                if start_line <= line <= end_line and chunk_id not in seen:
                    seen.add(chunk_id)
                    chunk_row = self.fts.get_chunk(chunk_id)
                    content = chunk_row["content"] if chunk_row else ""
                    lang = chunk_row["language"] if chunk_row else ""
                    results.append(
                        {
                            "chunk_id": chunk_id,
                            "id": chunk_id,
                            "file_id": file_id,
                            "language": lang,
                            "start_line": start_line,
                            "end_line": end_line,
                            "content": content,
                            "metadata": {
                                "document_id": file_id,
                                "filepath": file_id,
                                "start_line": start_line,
                                "end_line": end_line,
                                "language": lang,
                                "chunk_type": "code",
                                "via_symbol": symbol_hit.get("name"),
                            },
                        }
                    )
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
