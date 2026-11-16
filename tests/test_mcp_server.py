"""MCP server tests through a real client: discovery, calls, threads, stdio."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp.client import Client
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from docstra.core.config.settings import UserConfig
from docstra.core.mcp.server import build_server
from docstra.core.services.ingestion_service import IngestionService

TOOLS = {
    "lookup_symbol",
    "who_references",
    "file_summary",
    "search",
    "get_doc_page",
    "list_doc_pages",
}


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "helpers.py").write_text("def helper() -> int:\n    return 1\n")
    (root / "app.py").write_text(
        "import helpers\n\n\ndef main() -> int:\n    return helpers.helper()\n"
    )
    config = UserConfig()
    config.storage.persist_directory = str(root / ".docstra")
    assert IngestionService().build_core_index(str(root), config)
    (root / ".docstra" / "config.yaml").write_text(
        f"storage:\n  persist_directory: {root / '.docstra'}\n"
    )
    return root


def test_tool_discovery_and_calls_in_memory(repo: Path) -> None:
    server = build_server(str(repo))

    async def run() -> None:
        async with Client(server) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            assert set(tools) == TOOLS
            assert set(tools["search"].input_schema["properties"]) == {
                "query",
                "n_results",
            }
            assert tools["search"].input_schema["required"] == ["query"]

            result = await client.call_tool("lookup_symbol", {"name": "helper"})
            assert not result.is_error
            definitions = result.structured_content["result"]["definitions"]
            assert any(d["filepath"] == "helpers.py" for d in definitions)

            result = await client.call_tool("who_references", {"filepath": "app.py"})
            assert "helpers.py" in result.structured_content["result"]["imports"]

            result = await client.call_tool("file_summary", {"filepath": "nope.py"})
            assert "error" in result.structured_content["result"]

            result = await client.call_tool("search", {"query": "helper"})
            assert result.structured_content["result"]["mode"] == "lexical"

            result = await client.call_tool("list_doc_pages", {})
            assert not result.is_error

            result = await client.call_tool("get_doc_page", {"page_path": "../x"})
            assert "Error" in result.content[0].text

    anyio.run(run)


def test_concurrent_searches_on_worker_threads(repo: Path) -> None:
    server = build_server(str(repo))
    results: list[Any] = []

    async def one(client: Client, query: str) -> None:
        results.append(await client.call_tool("search", {"query": query}))

    async def run() -> None:
        async with Client(server) as client:
            async with anyio.create_task_group() as tg:
                for i in range(24):
                    tg.start_soon(one, client, ["helper", "main"][i % 2])

    anyio.run(run)
    assert len(results) == 24
    assert all(not r.is_error for r in results)
    assert all(r.structured_content["result"]["mode"] == "lexical" for r in results)


def test_cli_stdio_transport(repo: Path) -> None:
    exe = shutil.which("docstra", path=str(Path(sys.executable).parent))
    assert exe, "docstra entry point not installed in this environment"
    params = StdioServerParameters(command=exe, args=["serve-mcp", str(repo)])

    async def run() -> None:
        with anyio.fail_after(60):
            async with stdio_client(params) as streams:
                async with ClientSession(*streams[:2]) as session:
                    await session.initialize()
                    tools = {t.name for t in (await session.list_tools()).tools}
                    assert tools == TOOLS
                    result = await session.call_tool("lookup_symbol", {"name": "main"})
                    assert not result.is_error
                    assert result.structured_content["result"]["symbol"] == "main"

    anyio.run(run)


def test_cli_reports_missing_index_on_stderr(tmp_path: Path) -> None:
    import subprocess

    exe = shutil.which("docstra", path=str(Path(sys.executable).parent))
    assert exe
    proc = subprocess.run(
        [exe, "serve-mcp", str(tmp_path)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert "No core index found" in proc.stderr
