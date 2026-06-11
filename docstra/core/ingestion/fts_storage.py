"""SQLite + FTS5 store for lexical retrieval over chunks and symbols."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from docstra.core.indexing.model import IndexedSymbol

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
        clauses = ["chunks_fts MATCH ?"]
        params: List[Any] = [query]
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
        return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    # --- symbols (filled in Task 2b) ---
