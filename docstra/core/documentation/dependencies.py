# File: ./docstra/core/documentation/dependencies.py
"""
Documentation dependency tracking system for incremental updates.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from docstra.core.document_processing.document import Document
from docstra.core.indexing.code_index import CodebaseIndex
from docstra.core.indexing.repo_map import RepositoryMap


@dataclass
class DocumentationDependency:
    """Tracks dependencies between documentation pages and source files."""
    doc_path: str                           # Generated doc file path
    source_files: List[str]                 # Source files this doc depends on
    related_docs: List[str]                 # Other docs that reference this
    dependency_hash: str                    # Hash of all source dependencies
    last_generated: float                   # Timestamp when doc was generated
    generation_context: Dict[str, Any]      # Context used for generation
    cross_references: List[Dict[str, str]]  # Cross-references to other files
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentationDependency:
        """Create from dictionary loaded from JSON."""
        return cls(**data)


class DocumentationDependencyTracker:
    """Manages dependencies between documentation and source code."""
    
    def __init__(self, persist_directory: str):
        self.persist_directory = Path(persist_directory)
        self.dependency_file = self.persist_directory / "doc_dependencies.json"
        self.dependencies: Dict[str, DocumentationDependency] = {}
        self._load_dependencies()
    
    def _load_dependencies(self) -> None:
        """Load existing dependencies from disk."""
        if self.dependency_file.exists():
            try:
                with open(self.dependency_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.dependencies = {
                        path: DocumentationDependency.from_dict(dep_data)
                        for path, dep_data in data.items()
                    }
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Warning: Could not load dependency file: {e}")
                self.dependencies = {}
    
    def _save_dependencies(self) -> None:
        """Save dependencies to disk."""
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {
                path: dep.to_dict()
                for path, dep in self.dependencies.items()
            }
            with open(self.dependency_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save dependencies: {e}")
    
    def track_documentation_dependencies(self, 
                                       doc_path: str, 
                                       document: Document,
                                       repo_map: Optional[RepositoryMap] = None,
                                       code_index: Optional[CodebaseIndex] = None) -> None:
        """Track all dependencies for a documentation page."""
        source_files = [document.metadata.filepath]
        cross_references = []
        
        # Add direct dependencies (imports, related files)
        if repo_map:
            dependencies = repo_map.get_file_dependencies(document.metadata.filepath)
            related_files = repo_map.get_related_files(document.metadata.filepath)
            source_files.extend(dependencies)
            source_files.extend(related_files)
            
            # Get cross-references from repo map
            if hasattr(repo_map, '_get_cross_references'):
                cross_references = repo_map._get_cross_references(document.metadata.filepath)
        
        # Add symbol-based dependencies via code index
        if code_index:
            for symbol in document.metadata.classes + document.metadata.functions:
                symbol_locations = code_index.search_symbol(symbol)
                symbol_files = [loc['filepath'] for loc in symbol_locations]
                source_files.extend(symbol_files)
                
                # Add cross-references from symbol usage
                for loc in symbol_locations:
                    if loc['filepath'] != document.metadata.filepath:
                        cross_references.append({
                            "file": loc['filepath'],
                            "symbol": symbol,
                            "type": "symbol_usage",
                            "line": str(loc.get('line', 'unknown'))
                        })
        
        # Remove duplicates while preserving order
        unique_source_files = []
        seen = set()
        for f in source_files:
            if f not in seen:
                unique_source_files.append(f)
                seen.add(f)
        
        # Calculate dependency hash for change detection
        dependency_hash = self._calculate_dependency_hash(unique_source_files)
        
        # Extract generation context
        generation_context = self._extract_generation_context(document, repo_map)
        
        self.dependencies[doc_path] = DocumentationDependency(
            doc_path=doc_path,
            source_files=unique_source_files,
            related_docs=[],  # Will be populated by cross-reference analysis
            dependency_hash=dependency_hash,
            last_generated=time.time(),
            generation_context=generation_context,
            cross_references=cross_references
        )
        
        self._save_dependencies()
    
    def get_impacted_documentation(self, changed_files: List[str]) -> List[str]:
        """Find all documentation pages that need regeneration due to file changes."""
        impacted_docs = []
        changed_files_set = set(changed_files)
        
        for doc_path, dep in self.dependencies.items():
            # Check if any source files of this doc were changed
            if any(source_file in changed_files_set for source_file in dep.source_files):
                impacted_docs.append(doc_path)
        
        return impacted_docs
    
    def needs_regeneration(self, doc_path: str) -> bool:
        """Check if a documentation page needs regeneration."""
        if doc_path not in self.dependencies:
            return True  # New doc, needs generation
        
        dep = self.dependencies[doc_path]
        current_hash = self._calculate_dependency_hash(dep.source_files)
        
        return current_hash != dep.dependency_hash
    
    def get_outdated_documentation(self) -> List[str]:
        """Get all documentation that is outdated based on source file changes."""
        outdated_docs = []
        
        for doc_path, dep in self.dependencies.items():
            if self.needs_regeneration(doc_path):
                outdated_docs.append(doc_path)
        
        return outdated_docs
    
    def update_cross_references(self) -> None:
        """Update cross-references between documentation pages."""
        # Build a map of which docs reference each source file
        file_to_docs: Dict[str, List[str]] = {}
        for doc_path, dep in self.dependencies.items():
            for source_file in dep.source_files:
                if source_file not in file_to_docs:
                    file_to_docs[source_file] = []
                file_to_docs[source_file].append(doc_path)
        
        # Update related_docs for each dependency
        for doc_path, dep in self.dependencies.items():
            related_docs = set()
            for source_file in dep.source_files:
                if source_file in file_to_docs:
                    for related_doc in file_to_docs[source_file]:
                        if related_doc != doc_path:
                            related_docs.add(related_doc)
            
            dep.related_docs = list(related_docs)
        
        self._save_dependencies()
    
    def _calculate_dependency_hash(self, source_files: List[str]) -> str:
        """Calculate hash of all source file dependencies for change detection."""
        hash_content = []
        
        for file_path in sorted(source_files):  # Sort for consistent hashing
            try:
                file_path_obj = Path(file_path)
                if file_path_obj.exists():
                    # Include file path and modification time
                    mtime = file_path_obj.stat().st_mtime
                    hash_content.append(f"{file_path}:{mtime}")
                else:
                    # File doesn't exist, include just the path
                    hash_content.append(f"{file_path}:missing")
            except Exception:
                # Handle permission errors, etc.
                hash_content.append(f"{file_path}:error")
        
        combined_content = "|".join(hash_content)
        return hashlib.sha256(combined_content.encode('utf-8')).hexdigest()
    
    def _extract_generation_context(self, 
                                   document: Document, 
                                   repo_map: Optional[RepositoryMap]) -> Dict[str, Any]:
        """Extract context information used during generation."""
        context = {
            "file_size": document.metadata.size_bytes,
            "line_count": document.metadata.line_count,
            "language": str(document.metadata.language),
            "classes": document.metadata.classes,
            "functions": document.metadata.functions,
            "imports": document.metadata.imports[:10],  # Limit for storage
            "last_modified": document.metadata.last_modified,
        }
        
        if repo_map:
            file_node = repo_map.find_file(document.metadata.filepath)
            if file_node:
                context.update({
                    "module_category": repo_map._categorize_module(document.metadata.filepath),
                    "complexity": file_node.complexity,
                    "dependencies_count": len(file_node.dependencies),
                    "dependents_count": len(file_node.dependents),
                })
        
        return context
    
    def get_dependency_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked dependencies."""
        if not self.dependencies:
            return {"total_docs": 0, "total_source_files": 0, "avg_dependencies": 0}
        
        total_docs = len(self.dependencies)
        all_source_files = set()
        total_deps = 0
        
        for dep in self.dependencies.values():
            all_source_files.update(dep.source_files)
            total_deps += len(dep.source_files)
        
        return {
            "total_docs": total_docs,
            "total_source_files": len(all_source_files),
            "avg_dependencies": total_deps / total_docs if total_docs > 0 else 0,
            "last_update": max(dep.last_generated for dep in self.dependencies.values()) if self.dependencies else 0
        }
    
    def cleanup_stale_dependencies(self, valid_doc_paths: Set[str]) -> int:
        """Remove dependencies for documentation that no longer exists."""
        stale_paths = []
        for doc_path in self.dependencies:
            if doc_path not in valid_doc_paths:
                stale_paths.append(doc_path)
        
        for path in stale_paths:
            del self.dependencies[path]
        
        if stale_paths:
            self._save_dependencies()
        
        return len(stale_paths) 