"""Analysis stage: organize documents into modules and gather statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from docstra.core.document_processing.document import Document
from docstra.core.indexing.repo_map import RepositoryMap


def module_category(filepath: str, repo_map: Optional[RepositoryMap] = None) -> str:
    """Return the module category for a file path."""
    if repo_map:
        return repo_map._categorize_module(filepath)
    return Path(filepath).parent.name or "root"


@dataclass
class CodebaseAnalysis:
    """Structured view of the documents to be documented."""

    documents: List[Document]
    module_structure: Dict[str, List[Document]]

    @property
    def total_files(self) -> int:
        return len(self.documents)

    @property
    def total_lines(self) -> int:
        return sum(doc.metadata.line_count for doc in self.documents)

    @property
    def languages(self) -> List[str]:
        return sorted({str(doc.metadata.language) for doc in self.documents})

    def documents_by_path(self) -> Dict[str, Document]:
        return {doc.metadata.filepath: doc for doc in self.documents}


def analyze_codebase(
    documents: List[Document], repo_map: Optional[RepositoryMap] = None
) -> CodebaseAnalysis:
    """Group documents by module category and collect basic statistics."""
    module_structure: Dict[str, List[Document]] = {}
    for doc in documents:
        category = module_category(doc.metadata.filepath, repo_map)
        module_structure.setdefault(category, []).append(doc)

    return CodebaseAnalysis(documents=documents, module_structure=module_structure)
