"""
Codebase indexing facade backed by the canonical core index manifest.
"""

from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

from docstra.core.document_processing.document import Document, DocumentType
from docstra.core.indexing.model import (
    CORE_INDEX_FILENAME,
    CoreIndexBuilder,
    CoreIndexManifest,
    EmbeddingRef,
    IndexedFile,
    IndexedSymbol,
    ImportRecord,
    normalize_file_id,
    resolve_file_path,
)

LEGACY_INDEX_FILENAMES = [
    "symbol_index.json",
    "file_index.json",
    "import_index.json",
    "function_index.json",
    "class_index.json",
]

RecordT = TypeVar("RecordT")


class CodebaseIndex:
    """Index for efficient search and retrieval of code elements."""

    def __init__(
        self,
        index_directory: str = ".docstra/index",
        codebase_root: Optional[str] = None,
    ):
        self.index_directory = index_directory
        self.codebase_root = (
            str(Path(codebase_root).resolve()) if codebase_root else None
        )
        self.manifest_path = Path(index_directory) / CORE_INDEX_FILENAME

        os.makedirs(index_directory, exist_ok=True)

        self._manifest = CoreIndexManifest.empty()
        self._files_by_id: Dict[str, IndexedFile] = {}
        self._symbols_by_name: Dict[str, List[IndexedSymbol]] = defaultdict(list)
        self._functions_by_name: Dict[str, List[IndexedSymbol]] = defaultdict(list)
        self._classes_by_name: Dict[str, List[IndexedSymbol]] = defaultdict(list)
        self._symbols_by_file: Dict[str, List[IndexedSymbol]] = defaultdict(list)
        self._imports_by_source: Dict[str, List[ImportRecord]] = defaultdict(list)
        self._imports_by_text: Dict[str, List[str]] = defaultdict(list)
        self._dependencies_by_source: Dict[str, List[str]] = defaultdict(list)
        self._dependents_by_target: Dict[str, List[str]] = defaultdict(list)

        self._load_manifest()

    @property
    def manifest(self) -> CoreIndexManifest:
        """Expose the loaded manifest."""
        return self._manifest

    @property
    def has_manifest(self) -> bool:
        """Return whether a persisted manifest is present."""
        return self.manifest_path.exists()

    @staticmethod
    def legacy_artifacts_in(index_directory: str | Path) -> List[Path]:
        """Return legacy sidecar index files that still exist in an index directory."""
        base = Path(index_directory)
        return [
            base / name for name in LEGACY_INDEX_FILENAMES if (base / name).exists()
        ]

    def _load_manifest(self) -> None:
        """Load the persisted manifest or detect legacy artifacts."""
        if self.manifest_path.exists():
            self._manifest = CoreIndexManifest.model_validate_json(
                self.manifest_path.read_text(encoding="utf-8")
            )
            self._rebuild_lookups()
            return

        legacy_paths = self.legacy_artifacts_in(self.index_directory)
        if legacy_paths:
            legacy_names = ", ".join(path.name for path in legacy_paths)
            raise FileNotFoundError(
                "Legacy Docstra index artifacts were found without a core index "
                f"manifest ({legacy_names}). Rerun 'docstra ingest' to rebuild the "
                "index in the new format."
            )

        self._manifest = CoreIndexManifest.empty()
        self._rebuild_lookups()

    def _rebuild_lookups(self) -> None:
        """Rebuild in-memory lookup tables from the manifest."""
        self._files_by_id = {
            indexed_file.id: indexed_file for indexed_file in self._manifest.files
        }
        self._symbols_by_name = defaultdict(list)
        self._functions_by_name = defaultdict(list)
        self._classes_by_name = defaultdict(list)
        self._symbols_by_file = defaultdict(list)
        self._imports_by_source = defaultdict(list)
        self._imports_by_text = defaultdict(list)
        self._dependencies_by_source = defaultdict(list)
        self._dependents_by_target = defaultdict(list)

        for symbol in self._manifest.symbols:
            self._symbols_by_name[symbol.name].append(symbol)
            self._symbols_by_file[symbol.file_id].append(symbol)
            if symbol.kind == "function":
                self._functions_by_name[symbol.name].append(symbol)
            elif symbol.kind == "class":
                self._classes_by_name[symbol.name].append(symbol)

        for import_record in self._manifest.imports:
            self._imports_by_source[import_record.source_file_id].append(import_record)
            self._imports_by_text[import_record.raw_text].append(
                import_record.source_file_id
            )

        for edge in self._manifest.edges:
            if edge.edge_type != "imports":
                continue
            self._dependencies_by_source[edge.source_id].append(edge.target_id)
            self._dependents_by_target[edge.target_id].append(edge.source_id)

    def replace_manifest(
        self, manifest: CoreIndexManifest, *, codebase_root: Optional[str] = None
    ) -> None:
        """Replace the in-memory manifest and rebuild lookup tables."""
        self._manifest = manifest
        if codebase_root is not None:
            self.codebase_root = str(Path(codebase_root).resolve())
        self._rebuild_lookups()

    def save(self) -> None:
        """Persist the current manifest."""
        self.manifest_path.write_text(
            self._manifest.model_dump_json(indent=2), encoding="utf-8"
        )

    def normalize_file_id(self, filepath: str) -> str:
        """Normalize a path or id to the canonical file id shape."""
        return normalize_file_id(filepath, self.codebase_root)

    def resolve_file_path(self, filepath: str) -> Optional[Path]:
        """Resolve a canonical file id to an absolute path when possible."""
        normalized = self.normalize_file_id(filepath)
        return resolve_file_path(normalized, self.codebase_root)

    def iter_files(self) -> List[IndexedFile]:
        """Return all indexed files."""
        return list(self._manifest.files)

    def iter_file_ids(self) -> List[str]:
        """Return all indexed file ids."""
        return [indexed_file.id for indexed_file in self._manifest.files]

    def index_document(self, document: Document) -> None:
        """Merge a single indexed document into the persisted manifest."""
        self.upsert_documents([document])

    def index_documents(self, documents: List[Document]) -> None:
        """Index multiple documents into a canonical manifest."""
        if documents and self.codebase_root is None:
            absolute_paths = [
                str(Path(document.metadata.filepath).resolve())
                for document in documents
                if Path(document.metadata.filepath).is_absolute()
            ]
            if absolute_paths:
                self.codebase_root = os.path.commonpath(absolute_paths)

        manifest = CoreIndexBuilder.from_documents(
            documents,
            codebase_root=self.codebase_root or Path.cwd(),
            embedding_backend=self._manifest.embedding_backend,
            embedding_model=self._manifest.embedding_model,
            source_kinds=self._manifest.source_kinds,
        )
        self.replace_manifest(manifest)
        self.save()

    def upsert_documents(self, documents: List[Document]) -> None:
        """Merge one or more indexed documents into the existing manifest."""
        if not documents:
            return

        if self.codebase_root is None:
            absolute_paths = [
                str(Path(document.metadata.filepath).resolve())
                for document in documents
                if Path(document.metadata.filepath).is_absolute()
            ]
            if absolute_paths:
                self.codebase_root = os.path.commonpath(absolute_paths)

        updated_manifest = CoreIndexBuilder.from_documents(
            documents,
            codebase_root=self.codebase_root or Path.cwd(),
            embedding_backend=self._manifest.embedding_backend,
            embedding_model=self._manifest.embedding_model,
            source_kinds=self._manifest.source_kinds,
            known_files=self._manifest.files,
        )
        merged_manifest = self._merge_manifest(updated_manifest)
        self.replace_manifest(merged_manifest)
        self.save()

    def _merge_manifest(self, updated_manifest: CoreIndexManifest) -> CoreIndexManifest:
        """Replace manifest records for indexed files while preserving other files."""
        updated_file_ids = {indexed_file.id for indexed_file in updated_manifest.files}
        if not updated_file_ids:
            return self._manifest

        return CoreIndexManifest(
            schema_version=updated_manifest.schema_version,
            created_at=updated_manifest.created_at,
            embedding_backend=updated_manifest.embedding_backend,
            embedding_model=updated_manifest.embedding_model,
            source_kinds=updated_manifest.source_kinds,
            files=self._merge_records(
                self._manifest.files,
                updated_manifest.files,
                lambda item: item.id in updated_file_ids,
            ),
            chunks=self._merge_records(
                self._manifest.chunks,
                updated_manifest.chunks,
                lambda item: item.file_id in updated_file_ids,
            ),
            symbols=self._merge_records(
                self._manifest.symbols,
                updated_manifest.symbols,
                lambda item: item.file_id in updated_file_ids,
            ),
            occurrences=self._merge_records(
                self._manifest.occurrences,
                updated_manifest.occurrences,
                lambda item: item.file_id in updated_file_ids,
            ),
            imports=self._merge_records(
                self._manifest.imports,
                updated_manifest.imports,
                lambda item: item.source_file_id in updated_file_ids,
            ),
            edges=self._merge_records(
                self._manifest.edges,
                updated_manifest.edges,
                lambda item: item.source_id in updated_file_ids,
            ),
            embeddings=self._merge_records(
                self._manifest.embeddings,
                updated_manifest.embeddings,
                lambda item: self._embedding_targets_file(item, updated_file_ids),
            ),
            docs=self._merge_records(
                self._manifest.docs,
                updated_manifest.docs,
                lambda item: bool(updated_file_ids.intersection(item.source_file_ids)),
            ),
        )

    @staticmethod
    def _merge_records(
        existing_records: List[RecordT],
        updated_records: List[RecordT],
        should_replace: Callable[[RecordT], bool],
    ) -> List[RecordT]:
        return [
            *[item for item in existing_records if not should_replace(item)],
            *updated_records,
        ]

    @staticmethod
    def _embedding_targets_file(
        embedding: EmbeddingRef, updated_file_ids: set[str]
    ) -> bool:
        if embedding.target_id in updated_file_ids:
            return True
        for file_id in updated_file_ids:
            if embedding.target_id.startswith(f"{file_id}#"):
                return True
            if embedding.target_id.startswith(f"{file_id}::"):
                return True
        return False

    def search_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Search for symbol definitions in the codebase."""
        return [
            self._symbol_location_payload(item)
            for item in self._symbols_by_name.get(symbol, [])
        ]

    def search_function(self, function_name: str) -> List[Dict[str, Any]]:
        """Search for function definitions in the codebase."""
        return [
            self._symbol_location_payload(item)
            for item in self._functions_by_name.get(function_name, [])
        ]

    def search_class(self, class_name: str) -> List[Dict[str, Any]]:
        """Search for class definitions in the codebase."""
        return [
            self._symbol_location_payload(item)
            for item in self._classes_by_name.get(class_name, [])
        ]

    def _symbol_location_payload(self, symbol: IndexedSymbol) -> Dict[str, Any]:
        return {
            "filepath": symbol.file_id,
            "line": symbol.line,
            "language": symbol.language,
            "kind": symbol.kind,
            "symbol_id": symbol.id,
        }

    def get_files_by_language(self, language: Union[DocumentType, str]) -> List[str]:
        """Get all indexed files for a language."""
        language_str = str(language)
        return [
            indexed_file.id
            for indexed_file in self._manifest.files
            if indexed_file.language == language_str
        ]

    def get_file_metadata(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Get derived metadata for an indexed file."""
        file_id = self.normalize_file_id(filepath)
        indexed_file = self._files_by_id.get(file_id)
        if indexed_file is None:
            return None

        symbols = self._symbols_by_file.get(file_id, [])
        classes = [symbol.name for symbol in symbols if symbol.kind == "class"]
        functions = [symbol.name for symbol in symbols if symbol.kind == "function"]
        imports = list(
            dict.fromkeys(
                record.raw_text for record in self._imports_by_source.get(file_id, [])
            )
        )
        dependencies = self.get_file_dependencies(file_id)
        dependents = self.get_dependents(file_id)

        return {
            "filepath": file_id,
            "language": indexed_file.language,
            "size_bytes": indexed_file.size_bytes,
            "line_count": indexed_file.line_count,
            "last_modified": indexed_file.last_modified,
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "module_docstring": indexed_file.module_docstring,
            "dependencies": dependencies,
            "dependents": dependents,
            "complexity": len(dependencies) + len(symbols),
            "complexity_metrics": {},
            "code_quality": {},
            "documentation_coverage": None,
            "test_coverage": None,
            "category": None,
            "contributors": [],
            "tags": [],
        }

    def search_files_by_import(self, import_stmt: str) -> List[str]:
        """Find files that contain a matching import statement."""
        if import_stmt in self._imports_by_text:
            return list(dict.fromkeys(self._imports_by_text[import_stmt]))

        results: List[str] = []
        for raw_text, file_ids in self._imports_by_text.items():
            if import_stmt in raw_text:
                results.extend(file_ids)
        return list(dict.fromkeys(results))

    def full_text_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform a simple full-text search across indexed files."""
        results = []
        for file_id, metadata in (
            (indexed_file.id, self.get_file_metadata(indexed_file.id))
            for indexed_file in self._manifest.files
        ):
            if metadata is None:
                continue
            absolute_path = self.resolve_file_path(file_id)
            if absolute_path is None:
                continue
            try:
                content = absolute_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if query.lower() not in content.lower():
                continue

            matches = []
            for line_number, line in enumerate(content.splitlines(), start=1):
                if query.lower() in line.lower():
                    matches.append(
                        {"line_number": line_number, "line_content": line.strip()}
                    )

            if matches:
                results.append(
                    {
                        "filepath": file_id,
                        "language": metadata["language"],
                        "matches": matches,
                    }
                )
        return results

    def get_file_dependencies(self, filepath: str) -> List[str]:
        """Return resolved file dependencies for a file."""
        file_id = self.normalize_file_id(filepath)
        return list(dict.fromkeys(self._dependencies_by_source.get(file_id, [])))

    def get_dependents(self, filepath: str) -> List[str]:
        """Return files that depend on a given file."""
        file_id = self.normalize_file_id(filepath)
        return list(dict.fromkeys(self._dependents_by_target.get(file_id, [])))

    def get_related_files(self, filepath: str) -> List[str]:
        """Find files related to a given file."""
        file_id = self.normalize_file_id(filepath)
        metadata = self.get_file_metadata(file_id)
        if metadata is None:
            return []

        related_files: set[str] = set()
        related_files.update(self.get_file_dependencies(file_id))
        related_files.update(self.get_dependents(file_id))

        for import_stmt in metadata["imports"]:
            related_files.update(self.search_files_by_import(import_stmt))

        for symbol in self._symbols_by_file.get(file_id, []):
            for match in self._symbols_by_name.get(symbol.name, []):
                if match.file_id != file_id:
                    related_files.add(match.file_id)

        related_files.discard(file_id)
        return sorted(related_files)

    def chunks_for_file(self, file_id: str) -> List[Tuple[str, int, int]]:
        """Return (chunk_id, start_line, end_line) tuples for a file in line order."""
        matching = [
            (chunk.id, chunk.start_line, chunk.end_line)
            for chunk in self._manifest.chunks
            if chunk.file_id == file_id
        ]
        matching.sort(key=lambda tup: tup[1])
        return matching

    def file_language(self, file_id: str) -> Optional[str]:
        """Return the language recorded in the manifest for a file id, if any."""
        entry = self._files_by_id.get(file_id)
        return entry.language if entry else None

    def clear(self) -> None:
        """Clear the persisted manifest and in-memory lookups."""
        self._manifest = CoreIndexManifest.empty(
            embedding_backend=self._manifest.embedding_backend,
            embedding_model=self._manifest.embedding_model,
            source_kinds=self._manifest.source_kinds,
        )
        self._rebuild_lookups()
        self.save()


class CodebaseIndexer:
    """Index a codebase for efficient search and retrieval."""

    def __init__(
        self,
        index_directory: str = ".docstra/index",
        exclude_patterns: Optional[List[str]] = None,
        codebase_root: Optional[str] = None,
        embedding_backend: str = "chroma",
        embedding_model: str = "",
        source_kinds: Optional[Iterable[str]] = None,
    ):
        self.index = CodebaseIndex(
            index_directory=index_directory, codebase_root=codebase_root
        )
        self.exclude_patterns = exclude_patterns or [
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
        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model
        self.source_kinds = list(source_kinds or ["tree-sitter"])

    def should_exclude(self, path: str) -> bool:
        """Check if a path should be excluded from indexing."""
        path_norm = os.path.normpath(path)
        path_parts = set(Path(path_norm).parts)
        basename = os.path.basename(path_norm)
        for pattern in self.exclude_patterns:
            if pattern in path_parts or basename == pattern:
                return True
        return False

    def index_document(self, document: Document) -> None:
        """Merge a single document into the manifest."""
        filtered_documents = [
            document
            for document in [document]
            if not self.should_exclude(document.metadata.filepath)
        ]
        self.index.upsert_documents(filtered_documents)

    def index_documents(self, documents: List[Document]) -> None:
        """Index multiple documents into the manifest."""
        filtered_documents = [
            document
            for document in documents
            if not self.should_exclude(document.metadata.filepath)
        ]
        if filtered_documents and self.index.codebase_root is None:
            absolute_paths = [
                str(Path(document.metadata.filepath).resolve())
                for document in filtered_documents
                if Path(document.metadata.filepath).is_absolute()
            ]
            if absolute_paths:
                self.index.codebase_root = os.path.commonpath(absolute_paths)

        manifest = CoreIndexBuilder.from_documents(
            filtered_documents,
            codebase_root=self.index.codebase_root or Path.cwd(),
            embedding_backend=self.embedding_backend,
            embedding_model=self.embedding_model,
            source_kinds=self.source_kinds,
        )
        self.index.replace_manifest(manifest)
        self.index.save()

    def get_index(self) -> CodebaseIndex:
        """Get the underlying codebase index."""
        return self.index

    def get_manifest(self) -> CoreIndexManifest:
        """Return the in-memory manifest built during indexing."""
        return self.index.manifest
