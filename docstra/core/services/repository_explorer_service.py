"""
Service for exploring repository structure and relationships.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from docstra.core.config.settings import UserConfig
from docstra.core.indexing.code_index import CodebaseIndex, CodebaseIndexer
from docstra.core.indexing.repo_map import RepositoryMap
from docstra.core.utils.colors import Colors


class RepositoryExplorerService:
    """Service for exploring repository structure and relationships."""

    def __init__(self, user_config: UserConfig, console: Optional[Console] = None):
        """Initialize the repository explorer service.

        Args:
            user_config: User configuration
            console: Optional console for output
        """
        self.user_config = user_config
        self.console = console or Console()
        self.repo_map: Optional[RepositoryMap] = None
        self.code_index: Optional[CodebaseIndex] = None

    def _get_persist_directory(self, abs_path: Path) -> Path:
        """Get the persistence directory for the codebase.

        Args:
            abs_path: Absolute path to the codebase

        Returns:
            Path to persistence directory
        """
        persist_dir_name = self.user_config.storage.persist_directory
        if not Path(persist_dir_name).is_absolute():
            return abs_path / persist_dir_name
        return Path(persist_dir_name).resolve()

    def _load_components(self, codebase_path: str) -> None:
        """Load repository map and code index for the given codebase.

        Args:
            codebase_path: Path to the codebase

        Raises:
            ValueError: If components cannot be loaded
        """
        abs_path = Path(codebase_path).resolve()
        persist_dir = self._get_persist_directory(abs_path)

        # Load code index
        index_path = persist_dir / "index"
        if index_path.exists():
            indexer = CodebaseIndexer(index_directory=str(index_path))
            self.code_index = indexer.get_index()

        # Load repository map
        map_path = persist_dir / "repo_map.json"
        if map_path.exists():
            # Create a new repository map and load from the saved data
            self.repo_map = RepositoryMap(str(abs_path), self.code_index)
            if self.code_index:
                self.repo_map.build()  # Rebuild with current index

        if not self.repo_map or not self.code_index:
            raise ValueError(
                "Repository not fully indexed. Run 'docstra ingest' first."
            )

    def get_file_relationships(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive file relationship information.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing relationship information
        """
        self._load_components(os.path.dirname(file_path) or ".")

        if not self.repo_map or not self.code_index:
            raise ValueError("Repository components not loaded")

        # Normalize file path
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        # Get basic relationships from repository map
        dependencies = self.repo_map.get_file_dependencies(file_path)
        related_files = self.repo_map.get_related_files(file_path)

        # Get dependents by finding files that import this one
        dependents = self._get_file_dependents(file_path)

        # Get symbols from code index
        file_metadata = self.code_index.get_file_metadata(file_path)
        symbols = []
        if file_metadata:
            symbols = file_metadata.get("functions", []) + file_metadata.get(
                "classes", []
            )

        # Get context from repository map
        context = self.repo_map.get_documentation_context_for_file(file_path)

        relationships = {
            "file_path": file_path,
            "dependencies": dependencies,
            "dependents": dependents,
            "related_files": related_files,
            "symbols": symbols,
            "complexity_info": context.get("file_info", {}),
            "architectural_info": context.get("architectural_info", {}),
            "cross_references": self.repo_map.get_cross_references(file_path),
        }

        return relationships

    def _get_file_dependents(self, file_path: str) -> List[str]:
        """Get files that depend on the given file.

        Args:
            file_path: Path to the file

        Returns:
            List of dependent file paths
        """
        if not self.code_index:
            return []

        dependents: List[str] = []
        file_metadata = self.code_index.get_file_metadata(file_path)
        if not file_metadata:
            return dependents

        # Find files that import symbols from this file
        for symbol in file_metadata.get("functions", []) + file_metadata.get(
            "classes", []
        ):
            symbol_usages = self.code_index.search_symbol(symbol)
            for usage in symbol_usages:
                if usage["filepath"] != file_path:
                    dependents.append(usage["filepath"])

        return list(set(dependents))  # Remove duplicates

    def explore_structure(
        self, path: str, depth: int = 3, show_tree: bool = False
    ) -> Dict[str, Any]:
        """Explore repository structure.

        Args:
            path: Path to explore
            depth: Maximum depth to explore
            show_tree: Whether to show as tree structure

        Returns:
            Dictionary containing structure information
        """
        self._load_components(path)

        if not self.repo_map:
            raise ValueError("Repository map not available")

        structure_data = {
            "root_path": self.repo_map.root_path,
            "statistics": self.repo_map.stats,
            "modules": {},
            "files": [],
            "directories": [],
        }

        if show_tree:
            structure_data["tree"] = self._build_tree_structure(depth)
        else:
            structure_data["flat"] = self._build_flat_structure(depth)

        return structure_data

    def _build_tree_structure(self, depth: int) -> Dict[str, Any]:
        """Build tree structure representation.

        Args:
            depth: Maximum depth

        Returns:
            Tree structure data
        """
        if not self.repo_map:
            return {}

        def build_node_tree(node, current_depth: int = 0) -> Dict[str, Any]:
            if current_depth >= depth:
                return {"truncated": True}

            node_data = {
                "name": node.name,
                "path": node.path,
                "type": "file" if hasattr(node, "language") else "directory",
            }

            if hasattr(node, "children"):  # DirectoryNode
                node_data["children"] = {}
                for name, child in node.children.items():
                    node_data["children"][name] = build_node_tree(
                        child, current_depth + 1
                    )
            elif hasattr(node, "language"):  # FileNode
                node_data["language"] = node.language
                node_data["size"] = node.size
                node_data["symbols"] = len(node.symbols)

            return node_data

        return build_node_tree(self.repo_map.root)

    def _build_flat_structure(self, depth: int) -> Dict[str, Any]:
        """Build flat structure representation.

        Args:
            depth: Maximum depth

        Returns:
            Flat structure data
        """
        if not self.repo_map:
            return {}

        files = []
        directories = []

        def traverse_node(
            node, current_depth: int = 0, path_parts: Optional[List[str]] = None
        ):
            if path_parts is None:
                path_parts = []

            if current_depth >= depth:
                return

            if hasattr(node, "children"):  # DirectoryNode
                directories.append(
                    {
                        "name": node.name,
                        "path": node.path,
                        "depth": current_depth,
                        "children_count": len(node.children),
                    }
                )

                for name, child in node.children.items():
                    traverse_node(child, current_depth + 1, path_parts + [name])

            elif hasattr(node, "language"):  # FileNode
                files.append(
                    {
                        "name": node.name,
                        "path": node.path,
                        "language": node.language,
                        "size": node.size,
                        "symbols": len(node.symbols),
                        "complexity": node.complexity,
                        "depth": current_depth,
                    }
                )

        traverse_node(self.repo_map.root)

        return {"files": files, "directories": directories}

    def display_file_relationships(self, relationships: Dict[str, Any]) -> None:
        """Display file relationships in a formatted table.

        Args:
            relationships: Relationship data to display
        """
        file_path = relationships["file_path"]
        self.console.print(
            f"\n[{Colors.BOLD}]File Relationships for:[/] {os.path.basename(file_path)}"
        )

        # Dependencies table
        if relationships["dependencies"]:
            deps_table = Table(
                title="Dependencies", show_header=True, header_style="bold cyan"
            )
            deps_table.add_column("File", style="cyan")
            deps_table.add_column("Type", style="yellow")

            for dep in relationships["dependencies"]:
                deps_table.add_row(os.path.basename(dep), "Import")

            self.console.print(deps_table)

        # Dependents table
        if relationships["dependents"]:
            deps_table = Table(
                title="Dependents", show_header=True, header_style="bold green"
            )
            deps_table.add_column("File", style="green")
            deps_table.add_column("Type", style="yellow")

            for dep in relationships["dependents"]:
                deps_table.add_row(os.path.basename(dep), "Uses")

            self.console.print(deps_table)

        # Symbols table
        if relationships["symbols"]:
            symbols_table = Table(
                title="Symbols", show_header=True, header_style="bold magenta"
            )
            symbols_table.add_column("Symbol", style="magenta")

            for symbol in relationships["symbols"][:10]:  # Show first 10
                symbols_table.add_row(symbol)

            if len(relationships["symbols"]) > 10:
                symbols_table.add_row(
                    f"... and {len(relationships['symbols']) - 10} more"
                )

            self.console.print(symbols_table)

        # Complexity info
        complexity_info = relationships.get("complexity_info", {})
        if complexity_info:
            info_text = f"[{Colors.BOLD}]Complexity:[/] {complexity_info.get('complexity', 'N/A')}\n"
            info_text += (
                f"[{Colors.BOLD}]Size:[/] {complexity_info.get('size_kb', 0):.1f} KB\n"
            )
            info_text += f"[{Colors.BOLD}]Module Type:[/] {complexity_info.get('module_type', 'Unknown')}"

            self.console.print(Panel(info_text, title="File Information", expand=False))
