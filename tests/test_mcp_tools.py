"""Tests for the MCP index toolbox and llms.txt emission."""

from __future__ import annotations

from pathlib import Path

from docstra.core.config.settings import UserConfig
from docstra.core.documentation.pipeline import write_llms_txt
from docstra.core.mcp.tools import IndexToolbox
from docstra.core.services.ingestion_service import IngestionService

import pytest


@pytest.fixture()
def indexed_repo(tmp_path: Path) -> tuple[Path, UserConfig]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "helpers.py").write_text("def helper() -> int:\n    return 1\n")
    (repo / "app.py").write_text(
        "import helpers\n\n\ndef main() -> int:\n    return helpers.helper()\n"
    )
    config = UserConfig()
    config.storage.persist_directory = str(repo / ".docstra")
    config.documentation.output_dir = str(repo / "docs_out")
    assert IngestionService().build_core_index(str(repo), config)
    return repo, config


def test_lookup_symbol_and_references(indexed_repo) -> None:
    repo, config = indexed_repo
    toolbox = IndexToolbox(str(repo), config)

    definitions = toolbox.lookup_symbol("helper")["definitions"]
    assert any(d["filepath"] == "helpers.py" for d in definitions)

    refs = toolbox.who_references("app.py")
    assert refs["file"] == "app.py"
    assert "helpers.py" in refs["imports"]

    refs = toolbox.who_references("helpers.py")
    assert "app.py" in refs["imported_by"]


def test_file_summary(indexed_repo) -> None:
    repo, config = indexed_repo
    toolbox = IndexToolbox(str(repo), config)

    summary = toolbox.file_summary("helpers.py")
    assert summary["filepath"] == "helpers.py"
    assert "helper" in summary["functions"]
    assert "app.py" in summary["dependents"]

    missing = toolbox.file_summary("nope.py")
    assert "error" in missing


def test_lexical_search_works_without_embeddings(indexed_repo) -> None:
    repo, config = indexed_repo
    toolbox = IndexToolbox(str(repo), config)

    result = toolbox.search("helper", n_results=5)
    assert result["mode"] == "lexical"
    assert any(hit["file"] == "helpers.py" for hit in result["results"])


def test_doc_pages_are_served_and_traversal_is_blocked(indexed_repo) -> None:
    repo, config = indexed_repo
    docs_root = Path(config.documentation.output_dir) / "docs"
    (docs_root / "api").mkdir(parents=True)
    (docs_root / "api" / "app.py.md").write_text("# app.py docs\n")

    toolbox = IndexToolbox(str(repo), config)

    assert toolbox.list_doc_pages() == ["api/app.py.md"]
    assert "# app.py docs" in toolbox.get_doc_page("api/app.py.md")
    assert "Error" in toolbox.get_doc_page("../../secrets.txt")
    assert "Error" in toolbox.get_doc_page("api/missing.md")


def test_toolbox_requires_core_index(tmp_path: Path) -> None:
    repo = tmp_path / "empty"
    repo.mkdir()
    config = UserConfig()
    config.storage.persist_directory = str(repo / ".docstra")

    with pytest.raises(FileNotFoundError, match="docstra index"):
        IndexToolbox(str(repo), config)


def test_mcp_server_exposes_tools(indexed_repo) -> None:
    import anyio

    repo, config = indexed_repo
    config.save_to_file(str(Path(repo) / ".docstra" / "config.yaml"))

    from docstra.core.mcp.server import build_server

    server = build_server(str(repo))
    tools = anyio.run(server.list_tools)
    names = {tool.name for tool in tools}
    assert {
        "lookup_symbol",
        "who_references",
        "file_summary",
        "search",
        "get_doc_page",
        "list_doc_pages",
    } <= names


def test_write_llms_txt(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    docs = output_dir / "docs"
    (docs / "api").mkdir(parents=True)
    (docs / "guides").mkdir()
    (docs / "index.md").write_text("# My Project\n\nOverview.\n")
    (docs / "guides" / "getting-started.md").write_text("# Getting Started\n")
    (docs / "api" / "app.py.md").write_text("# app.py\n\nDetails.\n")

    llms_path, full_path = write_llms_txt(output_dir, "My Project", "A test project")

    llms = llms_path.read_text(encoding="utf-8")
    assert llms.startswith("# My Project")
    assert "> A test project" in llms
    assert "- [My Project](index.md)" in llms
    assert "- [Getting Started](guides/getting-started.md)" in llms
    assert "- [app.py](api/app.py.md)" in llms

    full = full_path.read_text(encoding="utf-8")
    assert "<!-- index.md -->" in full
    assert "Details." in full
