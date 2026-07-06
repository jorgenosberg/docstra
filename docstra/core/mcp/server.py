"""FastMCP server wiring for the Docstra index toolbox."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from docstra.core.config.settings import ConfigManager
from docstra.core.mcp.tools import IndexToolbox


def build_server(
    codebase_path: str = ".", config_path: Optional[str] = None
) -> FastMCP:
    """Build the MCP server for a codebase.

    Args:
        codebase_path: Path to the indexed codebase
        config_path: Optional path to the Docstra config file

    Returns:
        A FastMCP server ready to run over stdio
    """
    codebase = Path(codebase_path).resolve()
    effective_config_path = config_path or str(codebase / ".docstra" / "config.yaml")
    user_config = ConfigManager(config_path=effective_config_path).config

    toolbox = IndexToolbox(str(codebase), user_config)
    mcp = FastMCP(
        "docstra",
        instructions=(
            "Docstra exposes a precomputed, verified index of this codebase: "
            "symbols, import graph, hybrid search, and generated documentation. "
            "Prefer these tools over exploring the repository file by file."
        ),
    )

    @mcp.tool()
    def lookup_symbol(name: str) -> Dict[str, Any]:
        """Find where a class or function is defined, with file and line."""
        return toolbox.lookup_symbol(name)

    @mcp.tool()
    def who_references(filepath: str) -> Dict[str, Any]:
        """List files that import a file and files it imports (verified graph edges)."""
        return toolbox.who_references(filepath)

    @mcp.tool()
    def file_summary(filepath: str) -> Dict[str, Any]:
        """Get indexed metadata for a file: classes, functions, imports, dependents."""
        return toolbox.file_summary(filepath)

    @mcp.tool()
    def search(query: str, n_results: int = 10) -> Dict[str, Any]:
        """Search code by meaning and keywords (hybrid BM25 + embeddings when available)."""
        return toolbox.search(query, n_results)

    @mcp.tool()
    def get_doc_page(page_path: str) -> str:
        """Read a generated documentation page (docs-relative path, e.g. 'api/src/app.py.md')."""
        return toolbox.get_doc_page(page_path)

    @mcp.tool()
    def list_doc_pages() -> List[str]:
        """List all generated documentation pages."""
        return toolbox.list_doc_pages()

    return mcp
