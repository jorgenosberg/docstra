"""Tests for the embedding-free core index build path."""

from __future__ import annotations

import json
from pathlib import Path

from docstra.core.config.settings import UserConfig
from docstra.core.indexing.code_index import CodebaseIndex
from docstra.core.indexing.model import CORE_INDEX_FILENAME
from docstra.core.services.ingestion_service import IngestionService


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "helpers.py").write_text("def helper() -> int:\n    return 1\n")
    (repo / "app.py").write_text(
        "import helpers\n\n\ndef main() -> int:\n    return helpers.helper()\n"
    )
    return repo


def test_build_core_index_creates_manifest_without_embeddings(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    config = UserConfig()
    config.storage.persist_directory = str(repo / ".docstra")

    service = IngestionService()
    assert service.build_core_index(str(repo), config)

    index_dir = repo / ".docstra" / "index"
    manifest_path = index_dir / CORE_INDEX_FILENAME
    assert manifest_path.exists()
    # No embeddings should have been generated on this path.
    assert not (repo / ".docstra" / "chroma").exists()

    index = CodebaseIndex(index_directory=str(index_dir), codebase_root=str(repo))
    file_ids = set(index.iter_file_ids())
    assert {"app.py", "helpers.py"} <= file_ids

    cross_refs = index.get_file_cross_references("app.py")
    assert "helpers.py" in cross_refs["imports"]

    cross_refs = index.get_file_cross_references("helpers.py")
    assert "app.py" in cross_refs["imported_by"]


def test_build_core_index_exports_manifest_json(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    config = UserConfig()
    config.storage.persist_directory = str(repo / ".docstra")
    export_path = tmp_path / "export" / "manifest.json"

    service = IngestionService()
    assert service.build_core_index(str(repo), config, export_path=str(export_path))

    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported["schema_version"] == 1
    assert {entry["id"] for entry in exported["files"]} == {"app.py", "helpers.py"}
    assert any(edge["target_id"] == "helpers.py" for edge in exported["edges"])
