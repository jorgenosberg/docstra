"""
Typed core index models and builders.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import posixpath
from pathlib import Path, PurePosixPath
import re
from typing import Dict, Iterable, List, Literal, Optional, cast

from pydantic import BaseModel, Field

from docstra.core.document_processing.document import Document

CORE_INDEX_FILENAME = "core_index.json"
CORE_INDEX_SCHEMA_VERSION = 1


def normalize_file_id(path: str | Path, codebase_root: str | Path | None = None) -> str:
    """Normalize a source path to a repo-relative POSIX file id when possible."""
    path_str = str(path)
    candidate = Path(path_str).expanduser()

    if codebase_root is not None:
        root = Path(codebase_root).expanduser().resolve()
        try:
            if candidate.is_absolute():
                relative = candidate.resolve().relative_to(root)
            else:
                relative = PurePosixPath(path_str)
            normalized = PurePosixPath(str(relative)).as_posix()
            return _strip_relative_prefix(posixpath.normpath(normalized))
        except ValueError:
            pass

    if candidate.is_absolute():
        return PurePosixPath(candidate.as_posix()).as_posix()

    normalized = PurePosixPath(path_str).as_posix()
    return _strip_relative_prefix(posixpath.normpath(normalized))


def _strip_relative_prefix(path: str) -> str:
    """Drop a leading ./ while preserving ../ segments."""
    if path == ".":
        return ""
    if path.startswith("./"):
        return path[2:]
    return path


def resolve_file_path(
    file_id: str, codebase_root: str | Path | None = None
) -> Optional[Path]:
    """Resolve a file id to an absolute path when a codebase root is available."""
    candidate = Path(file_id).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if codebase_root is None:
        return None
    return (Path(codebase_root).expanduser().resolve() / file_id).resolve()


def make_chunk_id(file_id: str, start_line: int, end_line: int) -> str:
    """Build a stable chunk id from a file id and line span."""
    return f"{file_id}#L{start_line}-L{end_line}"


def make_symbol_id(file_id: str, kind: str, name: str, line: int) -> str:
    """Build a stable symbol id from a file id and symbol definition."""
    return f"{file_id}::{kind}::{name}::L{line}"


class IndexedFile(BaseModel):
    """Canonical file record."""

    id: str
    language: str
    size_bytes: int
    last_modified: float
    line_count: int
    module_docstring: Optional[str] = None


class IndexedChunk(BaseModel):
    """Canonical chunk record."""

    id: str
    file_id: str
    language: str
    start_line: int
    end_line: int
    chunk_type: str
    symbols: List[str] = Field(default_factory=list)
    parent_symbols: List[str] = Field(default_factory=list)


class IndexedSymbol(BaseModel):
    """Canonical symbol definition record."""

    id: str
    file_id: str
    name: str
    kind: Literal["class", "function", "symbol"]
    language: str
    line: int
    parent_symbols: List[str] = Field(default_factory=list)


class SymbolOccurrence(BaseModel):
    """Observed symbol location."""

    id: str
    symbol_id: str
    file_id: str
    start_line: int
    end_line: int
    occurrence_type: Literal["definition"] = "definition"


class ImportRecord(BaseModel):
    """Raw import statement with optional resolution."""

    id: str
    source_file_id: str
    raw_text: str
    target_file_id: Optional[str] = None


class CodeEdge(BaseModel):
    """Relationship between indexed entities."""

    id: str
    source_id: str
    target_id: str
    edge_type: Literal["imports"] = "imports"


class EmbeddingRef(BaseModel):
    """Reference to a stored vector in the embedding backend."""

    target_id: str
    target_kind: Literal["file", "chunk", "symbol"]
    backend: str
    collection_name: str
    vector_id: str


class GeneratedDoc(BaseModel):
    """Generated documentation artifact metadata."""

    id: str
    source_file_ids: List[str] = Field(default_factory=list)
    output_path: Optional[str] = None
    generated_at: Optional[datetime] = None


class CoreIndexManifest(BaseModel):
    """Canonical persisted code index manifest."""

    schema_version: int = CORE_INDEX_SCHEMA_VERSION
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    embedding_backend: str = "chroma"
    embedding_model: str = ""
    source_kinds: List[str] = Field(default_factory=lambda: ["tree-sitter"])
    files: List[IndexedFile] = Field(default_factory=list)
    chunks: List[IndexedChunk] = Field(default_factory=list)
    symbols: List[IndexedSymbol] = Field(default_factory=list)
    occurrences: List[SymbolOccurrence] = Field(default_factory=list)
    imports: List[ImportRecord] = Field(default_factory=list)
    edges: List[CodeEdge] = Field(default_factory=list)
    embeddings: List[EmbeddingRef] = Field(default_factory=list)
    docs: List[GeneratedDoc] = Field(default_factory=list)

    @classmethod
    def empty(
        cls,
        *,
        embedding_backend: str = "chroma",
        embedding_model: str = "",
        source_kinds: Optional[Iterable[str]] = None,
    ) -> CoreIndexManifest:
        """Create an empty manifest."""
        return cls(
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            source_kinds=list(source_kinds or ["tree-sitter"]),
        )


class CoreIndexBuilder:
    """Build a core manifest from processed documents."""

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        codebase_root: str | Path,
        *,
        embedding_backend: str = "chroma",
        embedding_model: str = "",
        source_kinds: Optional[Iterable[str]] = None,
        known_files: Optional[Iterable[IndexedFile]] = None,
    ) -> CoreIndexManifest:
        """Build a canonical manifest from processed documents."""
        root = Path(codebase_root).expanduser().resolve()
        manifest = CoreIndexManifest.empty(
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            source_kinds=source_kinds,
        )

        file_symbol_parents: Dict[str, Dict[str, List[str]]] = defaultdict(dict)

        for document in documents:
            file_id = normalize_file_id(document.metadata.filepath, root)
            manifest.files.append(
                IndexedFile(
                    id=file_id,
                    language=str(document.metadata.language),
                    size_bytes=document.metadata.size_bytes,
                    last_modified=document.metadata.last_modified,
                    line_count=document.metadata.line_count,
                    module_docstring=document.metadata.module_docstring,
                )
            )

            manifest.embeddings.append(
                EmbeddingRef(
                    target_id=file_id,
                    target_kind="file",
                    backend=embedding_backend,
                    collection_name="documents",
                    vector_id=file_id,
                )
            )

            for chunk in document.chunks:
                chunk_id = make_chunk_id(file_id, chunk.start_line, chunk.end_line)
                manifest.chunks.append(
                    IndexedChunk(
                        id=chunk_id,
                        file_id=file_id,
                        language=str(document.metadata.language),
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        chunk_type=chunk.chunk_type,
                        symbols=list(chunk.symbols),
                        parent_symbols=list(chunk.parent_symbols),
                    )
                )
                manifest.embeddings.append(
                    EmbeddingRef(
                        target_id=chunk_id,
                        target_kind="chunk",
                        backend=embedding_backend,
                        collection_name="chunks",
                        vector_id=chunk_id,
                    )
                )
                for symbol_name in chunk.symbols:
                    if symbol_name:
                        file_symbol_parents[file_id][symbol_name] = list(
                            chunk.parent_symbols
                        )

        available_files = cls._merge_available_files(
            known_files or [],
            manifest.files,
        )
        module_map = cls._build_python_module_map(available_files)
        file_id_set = {file.id for file in available_files}

        for document in documents:
            file_id = normalize_file_id(document.metadata.filepath, root)
            language = str(document.metadata.language)
            symbol_kind_map = cls._build_symbol_kind_map(document)

            for symbol_name, lines in document.metadata.symbols.items():
                for line in lines:
                    kind = symbol_kind_map.get(symbol_name, "symbol")
                    symbol_id = make_symbol_id(file_id, kind, symbol_name, line)
                    manifest.symbols.append(
                        IndexedSymbol(
                            id=symbol_id,
                            file_id=file_id,
                            name=symbol_name,
                            kind=cast(Literal["class", "function", "symbol"], kind),
                            language=language,
                            line=line,
                            parent_symbols=file_symbol_parents[file_id].get(
                                symbol_name, []
                            ),
                        )
                    )
                    manifest.occurrences.append(
                        SymbolOccurrence(
                            id=f"{symbol_id}::definition",
                            symbol_id=symbol_id,
                            file_id=file_id,
                            start_line=line,
                            end_line=line,
                        )
                    )

            for index, raw_import in enumerate(document.metadata.imports):
                target_file_ids = cls._resolve_import_targets(
                    source_file_id=file_id,
                    raw_import=raw_import,
                    language=language,
                    module_map=module_map,
                    file_id_set=file_id_set,
                )
                if not target_file_ids:
                    manifest.imports.append(
                        ImportRecord(
                            id=f"{file_id}::import::{index}",
                            source_file_id=file_id,
                            raw_text=raw_import,
                            target_file_id=None,
                        )
                    )
                    continue

                multiple_targets = len(target_file_ids) > 1
                for target_index, target_file_id in enumerate(target_file_ids):
                    import_record_id = f"{file_id}::import::{index}"
                    if multiple_targets:
                        import_record_id = f"{import_record_id}::{target_index}"
                    import_record = ImportRecord(
                        id=import_record_id,
                        source_file_id=file_id,
                        raw_text=raw_import,
                        target_file_id=target_file_id,
                    )
                    manifest.imports.append(import_record)
                    manifest.edges.append(
                        CodeEdge(
                            id=f"{import_record.id}::imports::{target_file_id}",
                            source_id=file_id,
                            target_id=target_file_id,
                        )
                    )

        return manifest

    @staticmethod
    def _build_symbol_kind_map(document: Document) -> Dict[str, str]:
        symbol_kind_map: Dict[str, str] = {}
        for class_name in document.metadata.classes:
            symbol_kind_map[class_name] = "class"
        for function_name in document.metadata.functions:
            symbol_kind_map[function_name] = "function"
        return symbol_kind_map

    @staticmethod
    def _build_python_module_map(files: List[IndexedFile]) -> Dict[str, str]:
        module_map: Dict[str, str] = {}
        for indexed_file in files:
            pure_path = PurePosixPath(indexed_file.id)
            if pure_path.suffix != ".py":
                continue
            stem_parts = list(pure_path.with_suffix("").parts)
            if stem_parts and stem_parts[-1] == "__init__":
                stem_parts = stem_parts[:-1]
            if not stem_parts:
                continue
            module_map[".".join(stem_parts)] = indexed_file.id
        return module_map

    @staticmethod
    def _merge_available_files(
        known_files: Iterable[IndexedFile],
        current_files: Iterable[IndexedFile],
    ) -> List[IndexedFile]:
        merged_files: Dict[str, IndexedFile] = {
            indexed_file.id: indexed_file for indexed_file in known_files
        }
        for indexed_file in current_files:
            merged_files[indexed_file.id] = indexed_file
        return list(merged_files.values())

    @classmethod
    def _resolve_import_targets(
        cls,
        *,
        source_file_id: str,
        raw_import: str,
        language: str,
        module_map: Dict[str, str],
        file_id_set: set[str],
    ) -> List[str]:
        if language == "python":
            return cls._resolve_python_import(raw_import, module_map)
        if language in {"javascript", "typescript"}:
            target_file_id = cls._resolve_js_import(
                source_file_id, raw_import, file_id_set
            )
            return [target_file_id] if target_file_id is not None else []
        return []

    @staticmethod
    def _resolve_python_import(
        raw_import: str, module_map: Dict[str, str]
    ) -> List[str]:
        import_match = re.match(r"^import\s+(.+)$", raw_import.strip())
        if import_match:
            resolved_targets: List[str] = []
            for module_spec in import_match.group(1).split(","):
                module_name = module_spec.strip().split(" as ")[0].strip()
                target = module_map.get(module_name)
                if target:
                    resolved_targets.append(target)
            return _unique_preserving_order(resolved_targets)

        from_match = re.match(
            r"^from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+)$", raw_import.strip()
        )
        if not from_match:
            return []

        module_name = from_match.group(1).strip()
        imported_names = [
            part.strip().split(" as ")[0].strip()
            for part in from_match.group(2).split(",")
        ]

        resolved_targets: List[str] = []
        for candidate in [f"{module_name}.{name}" for name in imported_names]:
            target = module_map.get(candidate)
            if target:
                resolved_targets.append(target)

        if resolved_targets:
            return _unique_preserving_order(resolved_targets)

        target = module_map.get(module_name)
        if target:
            return [target]
        return []

    @staticmethod
    def _resolve_js_import(
        source_file_id: str, raw_import: str, file_id_set: set[str]
    ) -> Optional[str]:
        match = re.search(r"""(?:from|require\()\s*['"]([^'"]+)['"]""", raw_import)
        if not match:
            return None

        specifier = match.group(1)
        if not specifier.startswith("."):
            return None

        source_dir = PurePosixPath(source_file_id).parent
        base_candidate = posixpath.normpath(str(source_dir / specifier))
        candidates = [
            base_candidate,
            f"{base_candidate}.js",
            f"{base_candidate}.jsx",
            f"{base_candidate}.ts",
            f"{base_candidate}.tsx",
            f"{base_candidate}/index.js",
            f"{base_candidate}/index.ts",
        ]
        for candidate in candidates:
            normalized = _strip_relative_prefix(candidate)
            if normalized in file_id_set:
                return normalized
        return None


def _unique_preserving_order(values: Iterable[str]) -> List[str]:
    """Return unique strings while preserving input order."""
    return list(dict.fromkeys(values))
