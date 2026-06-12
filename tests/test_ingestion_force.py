"""Verify that a forced reindex clears the legacy FTS database file."""

from pathlib import Path

from docstra.core.config.settings import UserConfig
from docstra.core.services.ingestion_service import IngestionService


def test_force_reindex_removes_index_db(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "file.py").write_text("def hello(): return 1\n")

    persist_dir = repo / ".docstra"
    persist_dir.mkdir()
    stale_db = persist_dir / "index.db"
    stale_db.write_bytes(b"stale-bytes-not-a-real-sqlite-file")
    assert stale_db.exists()

    config = UserConfig()
    config.storage.persist_directory = str(persist_dir)
    service = IngestionService()
    service.ingest_codebase(str(repo), config, force=True)

    # The forced reindex should have removed the stale FTS DB before rebuilding;
    # the new one created by ingestion will be a valid SQLite file.
    assert stale_db.exists(), "expected ingestion to recreate the FTS DB"
    # And it should not be the stale bytes we put down.
    assert stale_db.read_bytes()[:16] != b"stale-bytes-not-"
