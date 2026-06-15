"""SQLite + FTS5 store for lexical retrieval over chunks and symbols."""

from __future__ import annotations

import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Sequence

from docstra.core.indexing.model import IndexedSymbol

_FTS_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _sanitize_fts_query(query: str) -> str:
    """Strip FTS5 syntax characters and lowercase the result.

    Tokenizes on word boundaries, joins with spaces, lowercases. The
    lowercasing matters: FTS5 boolean operators (NOT/AND/OR) are recognized
    only in uppercase, so lowercasing user queries prevents accidental
    boolean parsing while leaving the unicode61 tokenizer's case-insensitive
    matching unchanged.
    """
    tokens = _FTS_TOKEN_RE.findall(query)
    return " ".join(tokens).lower()

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    file_id    TEXT NOT NULL,
    language   TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    content    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    symbol_id UNINDEXED,
    file_id   UNINDEXED,
    kind      UNINDEXED,
    name,
    tokenize='unicode61 remove_diacritics 2'
);
"""


class FtsStorage:
    """SQLite store with FTS5 indexes for chunks and symbols."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        with self._conn:
            self._conn.executescript(_SCHEMA)
            current = self._conn.execute("SELECT version FROM schema_version").fetchone()
            if current is None:
                self._conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
                )

    def close(self) -> None:
        self._conn.close()

    # --- chunks ---

    def add_chunks(
        self,
        *,
        chunk_ids: Sequence[str],
        file_ids: Sequence[str],
        languages: Sequence[str],
        start_lines: Sequence[int],
        end_lines: Sequence[int],
        contents: Sequence[str],
    ) -> None:
        rows = list(zip(chunk_ids, file_ids, languages, start_lines, end_lines, contents))
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO chunks (chunk_id, file_id, language, start_line, end_line, content)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    file_id=excluded.file_id,
                    language=excluded.language,
                    start_line=excluded.start_line,
                    end_line=excluded.end_line,
                    content=excluded.content
                """,
                rows,
            )

    def delete_by_file(self, file_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            self._conn.execute("DELETE FROM symbols_fts WHERE file_id = ?", (file_id,))

    def search_chunks(
        self, query: str, n_results: int = 50, *, language: Optional[str] = None, file_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        match_query = _sanitize_fts_query(query)
        if not match_query:
            return []
        clauses = ["chunks_fts MATCH ?"]
        params: List[Any] = [match_query]
        if language is not None:
            clauses.append("chunks.language = ?")
            params.append(language)
        if file_id is not None:
            clauses.append("chunks.file_id = ?")
            params.append(file_id)
        sql = f"""
            SELECT chunks.chunk_id, chunks.file_id, chunks.language,
                   chunks.start_line, chunks.end_line, chunks.content,
                   -bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks ON chunks.rowid = chunks_fts.rowid
            WHERE {' AND '.join(clauses)}
            ORDER BY score DESC
            LIMIT ?
        """
        params.append(n_results)
        results = []
        for row in self._conn.execute(sql, params).fetchall():
            results.append({
                "id": row["chunk_id"],
                "chunk_id": row["chunk_id"],
                "file_id": row["file_id"],
                "language": row["language"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "content": row["content"],
                "score": row["score"],
                "metadata": {
                    "document_id": row["file_id"],
                    "filepath": row["file_id"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "language": row["language"],
                    "chunk_type": "code",
                },
            })
        return results

    # --- symbols ---

    def add_symbols(self, symbols: List[IndexedSymbol]) -> None:
        rows = [
            (symbol.id, symbol.file_id, symbol.kind, symbol.name) for symbol in symbols
        ]
        with self._conn:
            self._conn.executemany(
                "INSERT INTO symbols_fts (symbol_id, file_id, kind, name) VALUES (?, ?, ?, ?)",
                rows,
            )

    def search_symbols(self, query: str, n_results: int = 25) -> List[Dict[str, Any]]:
        match_query = _sanitize_fts_query(query)
        if not match_query:
            return []
        sql = """
            SELECT symbol_id, file_id, kind, name, -bm25(symbols_fts) AS score
            FROM symbols_fts
            WHERE symbols_fts MATCH ?
            ORDER BY score DESC
            LIMIT ?
        """
        results = []
        for row in self._conn.execute(sql, (match_query, n_results)).fetchall():
            results.append({
                "id": row["symbol_id"],
                "symbol_id": row["symbol_id"],
                "file_id": row["file_id"],
                "kind": row["kind"],
                "name": row["name"],
                "score": row["score"],
                "metadata": {
                    "document_id": row["file_id"],
                    "filepath": row["file_id"],
                    "name": row["name"],
                    "kind": row["kind"],
                },
            })
        return results
