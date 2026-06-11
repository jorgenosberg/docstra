"""
Repository mapping for understanding codebase structure.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Union, cast

from docstra.core.document_processing.document import Document
from docstra.core.indexing.code_index import CodebaseIndex
from docstra.core.indexing.model import normalize_file_id


class FileNode:
    """Node representing a file in the repository structure."""

    def __init__(self, name: str, path: str, language: Optional[str] = None):
        self.name = name
        self.path = path
        self.language = language
        self.size: Optional[int] = None
        self.symbols: List[str] = []
        self.imports: List[str] = []
        self.line_count: Optional[int] = None
        self.complexity: Optional[int] = None
        self.dependencies: List[str] = []
        self.dependents: List[str] = []
        self.category: Optional[str] = None
        self.last_modified: Optional[float] = None
        self.contributors: List[str] = []
        self.tags: List[str] = []
        self.analysis: Dict[str, Any] = {
            "complexity_metrics": {},
            "code_quality": {},
            "documentation_coverage": None,
            "test_coverage": None,
        }

    def analyze(self, index: Optional[CodebaseIndex] = None) -> None:
        """Analyze the file for additional metadata."""
        if not index:
            return

        metadata = index.get_file_metadata(self.path)
        if metadata is None:
            return

        self.line_count = metadata.get("line_count")
        self.complexity = metadata.get("complexity")
        self.dependencies = metadata.get("dependencies", [])
        self.dependents = metadata.get("dependents", [])
        self.category = metadata.get("category")
        self.last_modified = metadata.get("last_modified")
        self.contributors = metadata.get("contributors", [])
        self.tags = metadata.get("tags", [])
        self.analysis.update(
            {
                "complexity_metrics": metadata.get("complexity_metrics", {}),
                "code_quality": metadata.get("code_quality", {}),
                "documentation_coverage": metadata.get("documentation_coverage"),
                "test_coverage": metadata.get("test_coverage"),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": "file",
            "name": self.name,
            "path": self.path,
            "language": self.language,
            "size": self.size,
            "symbols": self.symbols,
            "imports": self.imports,
            "line_count": self.line_count,
            "complexity": self.complexity,
            "dependencies": self.dependencies,
            "dependents": self.dependents,
            "category": self.category,
            "last_modified": self.last_modified,
            "contributors": self.contributors,
            "tags": self.tags,
            "analysis": self.analysis,
        }


class DirectoryNode:
    """Node representing a directory in the repository structure."""

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.children: Dict[str, Union[FileNode, DirectoryNode]] = {}

    def add_file(self, file_path: str, language: Optional[str] = None) -> FileNode:
        file_name = os.path.basename(file_path)
        node = FileNode(file_name, file_path, language)
        self.children[file_name] = node
        return node

    def add_directory(self, dir_path: str) -> DirectoryNode:
        dir_name = os.path.basename(dir_path)
        node = DirectoryNode(dir_name, dir_path)
        self.children[dir_name] = node
        return node

    def get_or_create_directory(self, dir_path: str) -> DirectoryNode:
        dir_name = os.path.basename(dir_path)
        if dir_name in self.children and isinstance(
            self.children[dir_name], DirectoryNode
        ):
            return cast(DirectoryNode, self.children[dir_name])
        return self.add_directory(dir_path)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "directory",
            "name": self.name,
            "path": self.path,
            "children": {
                name: child.to_dict() for name, child in sorted(self.children.items())
            },
        }


class RepositoryMap:
    """Map representing the structure of a code repository."""

    def __init__(self, root_path: str, index: Optional[CodebaseIndex] = None):
        self.root_path = os.path.normpath(root_path)
        self.root = DirectoryNode(os.path.basename(self.root_path), self.root_path)
        self.index = index
        self.exclude_patterns: List[str] = [
            ".git",
            "__pycache__",
            ".mypy_cache",
            ".ruff_cache",
            ".pytest_cache",
            ".coverage",
            ".tox",
            ".nox",
            "node_modules",
            "venv",
            ".venv",
            "env",
            ".env",
            ".vscode",
            ".idea",
            "build",
            "dist",
        ]
        self.module_categories: Dict[str, List[str]] = {
            "core": ["core", "src", "lib", "main"],
            "api": ["api", "rest", "graphql", "endpoints"],
            "models": ["models", "schemas", "entities"],
            "services": ["services", "providers", "managers"],
            "utils": ["utils", "helpers", "common"],
            "tests": ["tests", "specs", "fixtures"],
            "config": ["config", "settings", "conf"],
            "docs": ["docs", "documentation"],
        }
        self.stats: Dict[str, Any] = {
            "total_files": 0,
            "total_lines": 0,
            "languages": {},
            "module_sizes": {},
            "dependencies": {},
            "complexity": {},
        }

    def should_exclude(self, path: str) -> bool:
        """Check if a path should be excluded based on exclude patterns."""
        path_norm = os.path.normpath(path)
        path_parts = set(Path(path_norm).parts)
        basename = os.path.basename(path_norm)
        for pattern in self.exclude_patterns:
            if pattern in path_parts or basename == pattern:
                return True
        return False

    def _categorize_module(self, path: str) -> str:
        """Categorize a module based on its path and contents."""
        path_lower = path.lower()
        for category, patterns in self.module_categories.items():
            if any(pattern in path_lower for pattern in patterns):
                return category

        if self.index:
            metadata = self.index.get_file_metadata(path)
            if metadata:
                if any(
                    test in path_lower for test in ["test_", "_test", "spec_", "_spec"]
                ):
                    return "tests"
                if any(
                    conf in path_lower
                    for conf in [".conf", ".config", ".yaml", ".yml", ".json"]
                ):
                    return "config"
                if path_lower.endswith((".md", ".rst", ".txt")):
                    return "docs"
        return "other"

    def _reset(self) -> None:
        self.root = DirectoryNode(os.path.basename(self.root_path), self.root_path)
        self.stats = {
            "total_files": 0,
            "total_lines": 0,
            "languages": {},
            "module_sizes": {},
            "dependencies": {},
            "complexity": {},
        }

    def build(self) -> None:
        """Build the repository map from the index when available."""
        self._reset()

        if self.index and self.index.iter_files():
            self._build_from_index()
        else:
            self._traverse_directory(self.root_path, self.root)
            if self.index:
                self._enhance_with_index()

        self._calculate_statistics()
        self._analyze_dependencies()

    def _build_from_index(self) -> None:
        if not self.index:
            return

        for indexed_file in self.index.iter_files():
            file_id = indexed_file.id
            current = self.root
            parts = list(PurePosixPath(file_id).parts)
            for segment_count in range(1, len(parts)):
                dir_path = "/".join(parts[:segment_count])
                current = current.get_or_create_directory(dir_path)

            file_node = current.add_file(file_id, indexed_file.language)
            file_node.size = indexed_file.size_bytes
            metadata = self.index.get_file_metadata(file_id) or {}
            file_node.symbols = metadata.get("classes", []) + metadata.get(
                "functions", []
            )
            file_node.imports = metadata.get("imports", [])

        self._enhance_with_index()

    def _traverse_directory(self, dir_path: str, node: DirectoryNode) -> None:
        try:
            for entry in os.scandir(dir_path):
                if self.should_exclude(entry.path):
                    continue

                if entry.is_file():
                    file_node = node.add_file(entry.path)
                    _, ext = os.path.splitext(entry.name)
                    language = self._get_language_from_extension(ext)
                    if language:
                        file_node.language = language
                    file_node.size = entry.stat().st_size
                elif entry.is_dir():
                    dir_node = node.add_directory(entry.path)
                    self._traverse_directory(entry.path, dir_node)
        except Exception as error:
            print(f"Error traversing {dir_path}: {error}")

    def _get_language_from_extension(self, ext: str) -> Optional[str]:
        ext = ext.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".php": "php",
            ".rb": "ruby",
            ".swift": "swift",
            ".kt": "kotlin",
            ".md": "markdown",
            ".txt": "text",
            ".html": "html",
            ".css": "css",
            ".sql": "sql",
            ".json": "json",
            ".xml": "xml",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
        }
        return language_map.get(ext)

    def _enhance_with_index(self) -> None:
        if not self.index:
            return

        def enhance(node: Union[FileNode, DirectoryNode]) -> None:
            if isinstance(node, FileNode):
                node.analyze(self.index)
            else:
                for child in node.children.values():
                    enhance(child)

        enhance(self.root)

    def _calculate_statistics(self) -> None:
        def analyze(node: Union[FileNode, DirectoryNode]) -> None:
            if isinstance(node, FileNode):
                self.stats["total_files"] = cast(int, self.stats["total_files"]) + 1
                if node.language:
                    languages = cast(Dict[str, int], self.stats["languages"])
                    languages[node.language] = languages.get(node.language, 0) + 1
                module_category = self._categorize_module(node.path)
                module_sizes = cast(Dict[str, int], self.stats["module_sizes"])
                module_sizes[module_category] = module_sizes.get(module_category, 0) + 1
                if node.line_count is not None:
                    self.stats["total_lines"] = (
                        cast(int, self.stats["total_lines"]) + node.line_count
                    )
            else:
                for child in node.children.values():
                    analyze(child)

        analyze(self.root)

    def _analyze_dependencies(self) -> None:
        def analyze(node: Union[FileNode, DirectoryNode]) -> None:
            if isinstance(node, FileNode):
                deps = self.get_file_dependencies(node.path)
                if deps:
                    dependencies = cast(
                        Dict[str, List[str]], self.stats["dependencies"]
                    )
                    dependencies[node.path] = deps
                complexity = len(deps) + len(node.symbols)
                complexity_dict = cast(Dict[str, int], self.stats["complexity"])
                complexity_dict[node.path] = complexity
            else:
                for child in node.children.values():
                    analyze(child)

        analyze(self.root)

    def _normalize_lookup_path(self, path: str) -> str:
        if self.index:
            return self.index.normalize_file_id(path)
        return normalize_file_id(path, self.root_path)

    def find_file(self, file_path: str) -> Optional[FileNode]:
        normalized = self._normalize_lookup_path(file_path)
        parts = normalized.split("/")
        current = self.root
        for part in parts[:-1]:
            child = current.children.get(part)
            if not isinstance(child, DirectoryNode):
                return None
            current = child
        leaf = current.children.get(parts[-1])
        if isinstance(leaf, FileNode):
            return leaf
        return None

    def find_directory(self, dir_path: str) -> Optional[DirectoryNode]:
        normalized = self._normalize_lookup_path(dir_path)
        if normalized in {"", "."}:
            return self.root

        parts = normalized.split("/")
        current = self.root
        for part in parts:
            child = current.children.get(part)
            if not isinstance(child, DirectoryNode):
                return None
            current = child
        return current

    def get_file_dependencies(self, file_path: str) -> List[str]:
        if not self.index:
            return []
        return self.index.get_file_dependencies(file_path)

    def get_related_files(self, file_path: str) -> List[str]:
        if not self.index:
            return []
        return self.index.get_related_files(file_path)

    def get_module_overview(self) -> Dict[str, Any]:
        overview = {
            "statistics": self.stats,
            "modules": {},
            "dependencies": {},
            "complexity": {},
        }

        def analyze(node: Union[FileNode, DirectoryNode]) -> None:
            if isinstance(node, FileNode):
                module_category = self._categorize_module(node.path)
                overview["modules"].setdefault(module_category, []).append(
                    {
                        "path": node.path,
                        "language": node.language,
                        "symbols": node.symbols,
                        "imports": node.imports,
                    }
                )
                if node.path in self.stats["dependencies"]:
                    overview["dependencies"][node.path] = self.stats["dependencies"][
                        node.path
                    ]
                if node.path in self.stats["complexity"]:
                    overview["complexity"][node.path] = self.stats["complexity"][
                        node.path
                    ]
            else:
                for child in node.children.values():
                    analyze(child)

        analyze(self.root)
        return overview

    def get_cross_references(self, file_path: str) -> List[Dict[str, str]]:
        file_id = self._normalize_lookup_path(file_path)
        cross_refs: List[Dict[str, str]] = []

        for dependency in self.get_file_dependencies(file_id):
            cross_refs.append(
                {
                    "file": dependency,
                    "type": "import",
                    "description": f"Imports from {os.path.basename(dependency)}",
                }
            )

        if self.index:
            for dependent in self.index.get_dependents(file_id):
                cross_refs.append(
                    {
                        "file": dependent,
                        "type": "imported_by",
                        "description": f"Used by {os.path.basename(dependent)}",
                    }
                )

        seen = {reference["file"] for reference in cross_refs}
        for related in self.get_related_files(file_id):
            if related == file_id or related in seen:
                continue
            cross_refs.append(
                {
                    "file": related,
                    "type": "related",
                    "description": f"Related file in same module: {os.path.basename(related)}",
                }
            )
        return cross_refs

    def get_change_impact_analysis(
        self, changed_files: List[str]
    ) -> Dict[str, List[str]]:
        impact_map: Dict[str, List[str]] = {}
        for file_path in changed_files:
            normalized = self._normalize_lookup_path(file_path)
            impacted_files = set(self.get_related_files(normalized))
            if self.index:
                impacted_files.update(self.index.get_dependents(normalized))
                for dependent in list(self.index.get_dependents(normalized)):
                    impacted_files.update(self.index.get_dependents(dependent))
            impact_map[normalized] = sorted(impacted_files)
        return impact_map

    def get_documentation_context_for_file(self, file_path: str) -> Dict[str, Any]:
        normalized = self._normalize_lookup_path(file_path)
        node = self.find_file(normalized)
        if not node:
            return {}

        return {
            "file_info": {
                "path": normalized,
                "module_type": self._categorize_module(normalized),
                "complexity": node.complexity,
                "size_kb": node.size / 1024 if node.size else 0,
            },
            "dependencies": {
                "direct_imports": node.dependencies,
                "import_count": len(node.dependencies),
            },
            "dependents": {
                "files_using_this": node.dependents,
                "dependent_count": len(node.dependents),
            },
            "relationships": {
                "related_files": self.get_related_files(normalized),
                "cross_references": self.get_cross_references(normalized),
                "module_category": self._categorize_module(normalized),
            },
            "architectural_info": {
                "is_core_module": len(node.dependents) > 3,
                "is_leaf_module": len(node.dependencies) == 0,
                "centrality_score": len(node.dependents) + len(node.dependencies),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        base_dict = self.root.to_dict()
        base_dict.update(
            {
                "statistics": self.stats,
                "module_overview": self.get_module_overview(),
            }
        )
        return base_dict

    @staticmethod
    def from_documents(
        documents: List[Document], root_path: str, index: Optional[CodebaseIndex] = None
    ) -> RepositoryMap:
        repo_map = RepositoryMap(root_path, index)

        for document in documents:
            file_id = normalize_file_id(document.metadata.filepath, root_path)
            current = repo_map.root
            parts = list(PurePosixPath(file_id).parts)
            for segment_count in range(1, len(parts)):
                dir_path = "/".join(parts[:segment_count])
                current = current.get_or_create_directory(dir_path)

            file_node = current.add_file(file_id, str(document.metadata.language))
            file_node.size = document.metadata.size_bytes
            file_node.symbols = document.metadata.classes + document.metadata.functions
            file_node.imports = document.metadata.imports

        if index:
            repo_map._enhance_with_index()
            repo_map._calculate_statistics()
            repo_map._analyze_dependencies()

        return repo_map
