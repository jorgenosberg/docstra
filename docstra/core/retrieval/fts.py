"""Retriever that delegates lexical search to FtsStorage."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from docstra.core.ingestion.fts_storage import FtsStorage


class FtsRetriever:
    """Thin wrapper exposing FtsStorage searches to higher-level retrievers."""

    def __init__(self, storage: FtsStorage) -> None:
        self.storage = storage

    def retrieve_chunks(
        self,
        query: str,
        n_results: int = 50,
        *,
        language: Optional[str] = None,
        file_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.storage.search_chunks(
            query, n_results=n_results, language=language, file_id=file_id
        )

    def retrieve_symbols(self, query: str, n_results: int = 25) -> List[Dict[str, Any]]:
        return self.storage.search_symbols(query, n_results=n_results)
