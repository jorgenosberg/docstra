# File: ./docstra/core/indexing/repo_map.py
"""
Repository mapping for understanding codebase structure.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union, cast

from docstra.core.document_processing.document import Document
from docstra.core.indexing.code_index import CodebaseIndex


class FileNode:
    """Node representing a file in the repository structure."""

    def __init__(self, name: str, path: str, language: Optional[str] = None):
        """Initialize a file node.

        Args:
            name: File name
            path: Full path to the file
            language: Programming language of the file
        """
        self.name = name
        self.path = path
        self.language = language
        self.size: Optional[int] = None
        self.symbols: List[str] = []
        self.imports: List[str] = []

        # Enhanced metadata
        self.line_count: Optional[int] = None
        self.complexity: Optional[int] = None
        self.dependencies: List[str] = []
        self.dependents: List[str] = []
        self.category: Optional[str] = None
        self.last_modified: Optional[float] = None
        self.contributors: List[str] = []
        self.tags: List[str] = []

        # Analysis results with explicit types
        self.analysis: Dict[str, Any] = {
            "complexity_metrics": {},
            "code_quality": {},
            "documentation_coverage": None,
            "test_coverage": None,
        }

    def analyze(self, index: Optional[CodebaseIndex] = None) -> None:
        """Analyze the file for additional metadata.

        Args:
            index: Optional codebase index for enhanced analysis
        """
        if not index:
            return

        # Get enhanced metadata from index
        metadata = index.get_file_metadata(self.path)
        if metadata:
            # Update basic metadata
            self.line_count = metadata.get("line_count")
            self.complexity = metadata.get("complexity")
            self.dependencies = metadata.get("dependencies", [])
            self.dependents = metadata.get("dependents", [])
            self.category = metadata.get("category")
            self.last_modified = metadata.get("last_modified")
            self.contributors = metadata.get("contributors", [])
            self.tags = metadata.get("tags", [])

            # Update analysis results
            self.analysis.update(
                {
                    "complexity_metrics": metadata.get("complexity_metrics", {}),
                    "code_quality": metadata.get("code_quality", {}),
                    "documentation_coverage": metadata.get("documentation_coverage"),
                    "test_coverage": metadata.get("test_coverage"),
                }
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of the node
        """
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
        """Initialize a directory node.

        Args:
            name: Directory name
            path: Full path to the directory
        """
        self.name = name
        self.path = path
        self.children: Dict[str, Union[FileNode, DirectoryNode]] = {}

    def add_file(self, file_path: str, language: Optional[str] = None) -> FileNode:
        """Add a file to this directory.

        Args:
            file_path: Path to the file
            language: Programming language of the file

        Returns:
            The created file node
        """
        file_name = os.path.basename(file_path)
        node = FileNode(file_name, file_path, language)
        self.children[file_name] = node
        return node

    def add_directory(self, dir_path: str) -> DirectoryNode:
        """Add a subdirectory to this directory.

        Args:
            dir_path: Path to the directory

        Returns:
            The created directory node
        """
        dir_name = os.path.basename(dir_path)
        node = DirectoryNode(dir_name, dir_path)
        self.children[dir_name] = node
        return node

    def get_or_create_directory(self, dir_path: str) -> DirectoryNode:
        """Get a directory node, creating it if it doesn't exist.

        Args:
            dir_path: Path to the directory

        Returns:
            The directory node
        """
        dir_name = os.path.basename(dir_path)

        if dir_name in self.children and isinstance(
            self.children[dir_name], DirectoryNode
        ):
            # Type is guaranteed by isinstance check - cast to ensure type checker knows
            return cast(DirectoryNode, self.children[dir_name])

        return self.add_directory(dir_path)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of the node
        """
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
        """Initialize the repository map.

        Args:
            root_path: Root path of the repository
            index: Optional codebase index for enhanced metadata
        """
        self.root_path = os.path.normpath(root_path)
        self.root = DirectoryNode(os.path.basename(root_path), self.root_path)
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

        # Enhanced metadata
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

        # Codebase statistics with explicit types
        self.stats: Dict[str, Any] = {
            "total_files": 0,
            "total_lines": 0,
            "languages": {},
            "module_sizes": {},
            "dependencies": {},
            "complexity": {},
        }

    def should_exclude(self, path: str) -> bool:
        """Check if a path should be excluded based on exclude patterns.

        Args:
            path: Path to check

        Returns:
            True if the path should be excluded, False otherwise
        """
        for pattern in self.exclude_patterns:
            if pattern in path:
                return True
        return False

    def _categorize_module(self, path: str) -> str:
        """Categorize a module based on its path and contents.

        Args:
            path: Path to the module

        Returns:
            Category name
        """
        path_lower = path.lower()

        # Check path against known categories
        for category, patterns in self.module_categories.items():
            if any(pattern in path_lower for pattern in patterns):
                return category

        # Check file contents for categorization
        if self.index:
            metadata = self.index.get_file_metadata(path)
            if metadata:
                # Check for test files
                if any(
                    test in path_lower for test in ["test_", "_test", "spec_", "_spec"]
                ):
                    return "tests"
                # Check for configuration files
                if any(
                    conf in path_lower
                    for conf in [".conf", ".config", ".yaml", ".yml", ".json"]
                ):
                    return "config"
                # Check for documentation
                if path_lower.endswith((".md", ".rst", ".txt")):
                    return "docs"

        return "other"

    def _analyze_dependencies(self) -> None:
        """Analyze dependencies between modules and files."""
        if not self.index:
            return

        def analyze_node(node: Union[FileNode, DirectoryNode]) -> None:
            if isinstance(node, FileNode):
                # Track file dependencies
                deps = self.get_file_dependencies(node.path)
                if deps:
                    # Use type cast to ensure proper typing
                    dependencies_dict = cast(
                        Dict[str, List[str]], self.stats["dependencies"]
                    )
                    dependencies_dict[node.path] = deps

                    # Calculate complexity based on dependencies and symbols
                    complexity = len(deps) + len(node.symbols)
                    complexity_dict = cast(Dict[str, int], self.stats["complexity"])
                    complexity_dict[node.path] = complexity

            elif isinstance(node, DirectoryNode):
                # Recursively analyze child nodes
                for child in node.children.values():
                    analyze_node(child)

        analyze_node(self.root)

    def _calculate_statistics(self) -> None:
        """Calculate codebase statistics."""

        def analyze_node(node: Union[FileNode, DirectoryNode]) -> None:
            if isinstance(node, FileNode):
                # Update file statistics
                self.stats["total_files"] = cast(int, self.stats["total_files"]) + 1

                # Track language statistics
                if node.language:
                    languages_dict = cast(Dict[str, int], self.stats["languages"])
                    languages_dict[node.language] = (
                        languages_dict.get(node.language, 0) + 1
                    )

                # Track module sizes
                module_category = self._categorize_module(node.path)
                module_sizes_dict = cast(Dict[str, int], self.stats["module_sizes"])
                module_sizes_dict[module_category] = (
                    module_sizes_dict.get(module_category, 0) + 1
                )

                # Count lines if available
                if node.line_count is not None:
                    self.stats["total_lines"] = (
                        cast(int, self.stats["total_lines"]) + node.line_count
                    )

            elif isinstance(node, DirectoryNode):
                # Recursively analyze child nodes
                for child in node.children.values():
                    analyze_node(child)

        analyze_node(self.root)

    def build(self) -> None:
        """Build the repository map by traversing the filesystem."""
        self._traverse_directory(self.root_path, self.root)

        # Enhance with metadata from the index if available
        if self.index:
            self._enhance_with_index()

        # Calculate statistics and analyze dependencies
        self._calculate_statistics()
        self._analyze_dependencies()

    def _traverse_directory(self, dir_path: str, node: DirectoryNode) -> None:
        """Recursively traverse a directory and build the map.

        Args:
            dir_path: Path to the directory
            node: Directory node representing the directory
        """
        try:
            for entry in os.scandir(dir_path):
                if self.should_exclude(entry.path):
                    continue

                if entry.is_file():
                    # Add file to the current directory node
                    file_node = node.add_file(entry.path)

                    # Determine language from file extension
                    _, ext = os.path.splitext(entry.name)
                    language = self._get_language_from_extension(ext)
                    if language:
                        file_node.language = language

                    # Set file size
                    file_node.size = entry.stat().st_size

                elif entry.is_dir():
                    # Add directory and recursively traverse it
                    dir_node = node.add_directory(entry.path)
                    self._traverse_directory(entry.path, dir_node)

        except Exception as e:
            # Handle permission errors and other issues
            print(f"Error traversing {dir_path}: {str(e)}")

    def _get_language_from_extension(self, ext: str) -> Optional[str]:
        """Determine programming language from file extension.

        Args:
            ext: File extension

        Returns:
            Language name if recognized, None otherwise
        """
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
        """Enhance the map with metadata from the codebase index."""
        if not self.index:
            return

        def _enhance_node(node: Union[FileNode, DirectoryNode]) -> None:
            """Recursively enhance nodes with index metadata."""
            if isinstance(node, FileNode):
                # Analyze file node
                node.analyze(self.index)

                # Update repository statistics
                if node.line_count is not None:
                    self.stats["total_lines"] = (
                        cast(int, self.stats["total_lines"]) + node.line_count
                    )

                if node.language:
                    languages_dict = cast(Dict[str, int], self.stats["languages"])
                    languages_dict[node.language] = (
                        languages_dict.get(node.language, 0) + 1
                    )

                if node.category:
                    module_sizes_dict = cast(Dict[str, int], self.stats["module_sizes"])
                    module_sizes_dict[node.category] = (
                        module_sizes_dict.get(node.category, 0) + 1
                    )

                # Update complexity metrics
                if node.complexity is not None:
                    complexity_dict = cast(Dict[str, int], self.stats["complexity"])
                    complexity_dict[node.path] = node.complexity

                # Update dependency information
                if node.dependencies:
                    dependencies_dict = cast(
                        Dict[str, List[str]], self.stats["dependencies"]
                    )
                    dependencies_dict[node.path] = node.dependencies

            elif isinstance(node, DirectoryNode):
                # Recursively enhance child nodes
                for child in node.children.values():
                    _enhance_node(child)

        # Start enhancement from the root
        _enhance_node(self.root)

    def find_file(self, file_path: str) -> Optional[FileNode]:
        """Find a file node by path.

        Args:
            file_path: Path to the file

        Returns:
            File node if found, None otherwise
        """
        file_path = os.path.normpath(file_path)

        # Find relative path from root
        rel_path = os.path.relpath(file_path, self.root_path)
        if rel_path.startswith(".."):
            # File is outside the repository
            return None

        parts = rel_path.split(os.sep)
        current = self.root

        # Navigate to parent directory
        for _i, part in enumerate(parts[:-1]):
            if part in current.children and isinstance(
                current.children[part], DirectoryNode
            ):
                current = cast(DirectoryNode, current.children[part])
            else:
                return None

        # Check if file exists in the directory
        file_name = parts[-1]
        if file_name in current.children and isinstance(
            current.children[file_name], FileNode
        ):
            return cast(FileNode, current.children[file_name])

        return None

    def find_directory(self, dir_path: str) -> Optional[DirectoryNode]:
        """Find a directory node by path.

        Args:
            dir_path: Path to the directory

        Returns:
            Directory node if found, None otherwise
        """
        dir_path = os.path.normpath(dir_path)

        # Find relative path from root
        rel_path = os.path.relpath(dir_path, self.root_path)
        if rel_path.startswith(".."):
            # Directory is outside the repository
            return None

        parts = rel_path.split(os.sep)
        if parts == ["."]:
            # Root directory
            return self.root

        current = self.root

        # Navigate to the directory
        for part in parts:
            if part in current.children and isinstance(
                current.children[part], DirectoryNode
            ):
                current = cast(DirectoryNode, current.children[part])
            else:
                return None

        return current

    def get_file_dependencies(self, file_path: str) -> List[str]:
        """Get dependencies of a file based on imports.

        Args:
            file_path: Path to the file

        Returns:
            List of file paths that are imported by the file
        """
        if not self.index:
            return []

        file_node = self.find_file(file_path)
        if not file_node:
            return []

        # Use index to find imported files
        imported_files = []
        for import_stmt in file_node.imports:
            # This is a simplified approach. A more sophisticated implementation
            # would resolve import statements to actual files.
            files = self.index.search_files_by_import(import_stmt)
            imported_files.extend(files)

        return imported_files

    def get_related_files(self, file_path: str) -> List[str]:
        """Get files related to a given file.

        Args:
            file_path: Path to the file

        Returns:
            List of related file paths
        """
        if not self.index:
            return []

        return self.index.get_related_files(file_path)

    def get_module_overview(self) -> Dict[str, Any]:
        """Get a comprehensive overview of the codebase modules.

        Returns:
            Dictionary containing module overview information
        """
        overview = {
            "statistics": self.stats,
            "modules": {},
            "dependencies": {},
            "complexity": {},
        }

        def analyze_node(node: Union[FileNode, DirectoryNode], path: str = "") -> None:
            if isinstance(node, FileNode):
                # Add file information
                module_category = self._categorize_module(node.path)
                if module_category not in overview["modules"]:
                    overview["modules"][module_category] = []

                file_info = {
                    "path": node.path,
                    "language": node.language,
                    "symbols": node.symbols,
                    "imports": node.imports,
                }
                overview["modules"][module_category].append(file_info)

                # Add dependency information
                if node.path in self.stats["dependencies"]:
                    overview["dependencies"][node.path] = self.stats["dependencies"][
                        node.path
                    ]

                # Add complexity information
                if node.path in self.stats["complexity"]:
                    overview["complexity"][node.path] = self.stats["complexity"][
                        node.path
                    ]

            elif isinstance(node, DirectoryNode):
                # Recursively analyze child nodes
                for name, child in node.children.items():
                    child_path = os.path.join(path, name)
                    analyze_node(child, child_path)

        analyze_node(self.root)
        return overview
    
    def get_cross_references(self, file_path: str) -> List[Dict[str, str]]:
        """Get cross-references for a file (imports, usage, etc.)."""
        cross_refs: List[Dict[str, str]] = []
        node = self.find_file(file_path)
        
        if not node:
            return cross_refs
        
        # Add imports as cross-references
        for import_path in node.dependencies:
            cross_refs.append({
                "file": import_path,
                "type": "import",
                "description": f"Imports from {os.path.basename(import_path)}"
            })
        
        # Add files that depend on this one
        for dependent_path in node.dependents:
            cross_refs.append({
                "file": dependent_path,
                "type": "imported_by",
                "description": f"Used by {os.path.basename(dependent_path)}"
            })
        
        # Add related files (same module/package)
        related_files = self.get_related_files(file_path)
        for related_path in related_files:
            if related_path != file_path and related_path not in [ref["file"] for ref in cross_refs]:
                cross_refs.append({
                    "file": related_path,
                    "type": "related",
                    "description": f"Related file in same module: {os.path.basename(related_path)}"
                })
        
        return cross_refs
    
    def get_change_impact_analysis(self, changed_files: List[str]) -> Dict[str, List[str]]:
        """Analyze the impact of changes to specific files."""
        impact_map = {}
        
        for file_path in changed_files:
            impacted_files = set()
            
            # Direct dependents (files that import this one)
            node = self.find_file(file_path)
            if node:
                impacted_files.update(node.dependents)
                
                # Indirect impact through dependency chain
                for dependent in node.dependents:
                    dependent_node = self.find_file(dependent)
                    if dependent_node:
                        impacted_files.update(dependent_node.dependents)
            
            # If no node found, try to find impact through symbol usage
            if not node and self.index:
                file_metadata = self.index.get_file_metadata(file_path)
                if file_metadata:
                    # Find files that use symbols from this file
                    for symbol in file_metadata.get('functions', []) + file_metadata.get('classes', []):
                        symbol_usages = self.index.search_symbol(symbol)
                        for usage in symbol_usages:
                            if usage['filepath'] != file_path:
                                impacted_files.add(usage['filepath'])
            
            impact_map[file_path] = list(impacted_files)
        
        return impact_map
    
    def get_documentation_context_for_file(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive context for documentation generation."""
        node = self.find_file(file_path)
        if not node:
            return {}
        
        context = {
            "file_info": {
                "path": file_path,
                "module_type": self._categorize_module(file_path),
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
                "related_files": self.get_related_files(file_path),
                "cross_references": self.get_cross_references(file_path),
                "module_category": self._categorize_module(file_path),
            },
            "architectural_info": {
                "is_core_module": len(node.dependents) > 3,  # Many files depend on it
                "is_leaf_module": len(node.dependencies) == 0,  # No dependencies
                "centrality_score": len(node.dependents) + len(node.dependencies),
            }
        }
        
        return context

    def to_dict(self) -> Dict[str, Any]:
        """Convert the repository map to a dictionary.

        Returns:
            Dictionary representation of the map
        """
        base_dict = self.root.to_dict()

        # Add enhanced metadata
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
        """Create a repository map from a list of documents.

        Args:
            documents: List of documents
            root_path: Root path of the repository
            index: Optional codebase index for enhanced metadata

        Returns:
            Repository map
        """
        repo_map = RepositoryMap(root_path, index)

        # Build directory structure
        for document in documents:
            file_path = document.metadata.filepath

            # Skip if outside root path
            if not os.path.commonpath([root_path, file_path]).startswith(root_path):
                continue

            # Get relative path from root
            rel_path = os.path.relpath(file_path, root_path)
            parts = rel_path.split(os.sep)

            current = repo_map.root

            # Create directories
            for i, _part in enumerate(parts[:-1]):
                dir_path = os.path.join(root_path, *parts[: i + 1])
                current = current.get_or_create_directory(dir_path)

            # Add file
            file_node = current.add_file(file_path, str(document.metadata.language))

            # Add metadata
            file_node.size = document.metadata.size_bytes
            file_node.symbols = document.metadata.classes + document.metadata.functions
            file_node.imports = document.metadata.imports

        return repo_map
