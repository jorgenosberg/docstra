"""Plain-Python tools over the Docstra index, wrapped by the MCP server.

Kept free of MCP imports and console output so the tools are testable in
isolation and safe for a stdio transport (which must not write to stdout).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from docstra.core.config.settings import UserConfig
from docstra.core.indexing.code_index import CodebaseIndex, CodebaseIndexer
from docstra.core.indexing.model import CORE_INDEX_FILENAME
from docstra.core.ingestion.fts_storage import FtsStorage
from docstra.core.retrieval.fts import FtsRetriever
from docstra.core.retrieval.fusion import FusionRetriever


class IndexToolbox:
    """Query surface over the core index, retrieval stack, and generated docs."""

    def __init__(self, codebase_path: str, user_config: UserConfig) -> None:
        self.codebase_path = Path(codebase_path).resolve()
        self.user_config = user_config

        persist_dir = Path(user_config.storage.persist_directory)
        if not persist_dir.is_absolute():
            persist_dir = self.codebase_path / persist_dir
        self.persist_dir = persist_dir.resolve()

        index_dir = self.persist_dir / "index"
        if not (index_dir / CORE_INDEX_FILENAME).exists():
            raise FileNotFoundError(
                f"No core index found at {index_dir}. Run 'docstra index' first."
            )
        self.code_index: CodebaseIndex = CodebaseIndexer(
            index_directory=str(index_dir),
            codebase_root=str(self.codebase_path),
        ).get_index()

        # Lexical retrieval works without embeddings; fusion needs Chroma.
        self.fts_retriever: Optional[FtsRetriever] = None
        self.fusion_retriever: Optional[FusionRetriever] = None
        index_db = self.persist_dir / "index.db"
        if index_db.exists():
            self.fts_retriever = FtsRetriever(FtsStorage(str(index_db)))

        chroma_check = self.persist_dir / "chroma" / "chroma.sqlite3"
        if chroma_check.exists() and self.fts_retriever:
            try:
                from docstra.core.ingestion.embeddings import EmbeddingFactory
                from docstra.core.ingestion.storage import ChromaDBStorage
                from docstra.core.retrieval.chroma import ChromaRetriever

                embedding_gen = EmbeddingFactory.create_embedding_generator(
                    embedding_type=user_config.embedding.provider,
                    model_name=user_config.embedding.model_name,
                    api_key=user_config.embedding.api_key or user_config.model.api_key,
                    api_base=user_config.model.api_base,
                )
                chroma_retriever = ChromaRetriever(
                    ChromaDBStorage(str(self.persist_dir / "chroma")),
                    embedding_gen,
                    codebase_root=str(self.codebase_path),
                )
                self.fusion_retriever = FusionRetriever(
                    dense=chroma_retriever,
                    fts=self.fts_retriever,
                    code_index=self.code_index,
                    rrf_k=user_config.retrieval.rrf_k,
                    fts_chunks_top_k=user_config.retrieval.fts_chunks_top_k,
                    fts_symbols_top_k=user_config.retrieval.fts_symbols_top_k,
                )
            except Exception:
                # Embeddings unavailable (no server, missing key): fall back
                # to lexical-only search rather than failing the whole server.
                self.fusion_retriever = None

        doc_config = user_config.documentation
        output_dir = Path(doc_config.output_dir if doc_config else "./docs")
        if not output_dir.is_absolute():
            output_dir = self.codebase_path / output_dir
        self.docs_root = (output_dir / "docs").resolve()
        if not self.docs_root.exists():
            self.docs_root = output_dir.resolve()

    def lookup_symbol(self, name: str) -> Dict[str, Any]:
        """Find definitions of a symbol (class or function) by name."""
        return {"symbol": name, "definitions": self.code_index.search_symbol(name)}

    def who_references(self, filepath: str) -> Dict[str, Any]:
        """Return graph-verified references for a file, keyed by direction."""
        file_id = self.code_index.normalize_file_id(filepath)
        refs = self.code_index.get_file_cross_references(file_id)
        return {"file": file_id, **refs}

    def file_summary(self, filepath: str) -> Dict[str, Any]:
        """Return indexed metadata for a file: symbols, imports, graph edges."""
        metadata = self.code_index.get_file_metadata(filepath)
        if metadata is None:
            return {
                "error": f"File not found in the index: {filepath}",
                "hint": "Paths are repo-relative, for example 'src/app.py'.",
            }
        return metadata

    def search(self, query: str, n_results: int = 10) -> Dict[str, Any]:
        """Search the codebase; hybrid when embeddings exist, lexical otherwise."""
        if self.fusion_retriever is None and self.fts_retriever is None:
            return {
                "error": "No searchable index found. Run 'docstra index' "
                "(lexical) or 'docstra ingest' (hybrid) first."
            }

        if self.fusion_retriever:
            chunks = self.fusion_retriever.retrieve_chunks(
                query=query, n_results=n_results
            )
            mode = "hybrid"
        else:
            # 'docstra index' alone populates symbols but not chunks, so
            # merge both lexical tables to stay useful without embeddings.
            assert self.fts_retriever is not None
            chunks = list(
                self.fts_retriever.retrieve_chunks(query=query, n_results=n_results)
            )
            for symbol in self.fts_retriever.retrieve_symbols(
                query, n_results=n_results
            ):
                chunks.append(
                    {
                        "metadata": symbol.get("metadata", {}),
                        "content": f"{symbol.get('kind', 'symbol')} "
                        f"{symbol.get('name', '')}",
                        "score": symbol.get("score"),
                    }
                )
            mode = "lexical"

        results = [
            {
                "file": chunk.get("metadata", {}).get("document_id", ""),
                "content": chunk.get("content", ""),
                "score": chunk.get("score"),
            }
            for chunk in chunks[:n_results]
        ]
        return {"query": query, "mode": mode, "results": results}

    def get_doc_page(self, page_path: str) -> str:
        """Read a generated documentation page by its docs-relative path."""
        candidate = (self.docs_root / page_path).resolve()
        try:
            candidate.relative_to(self.docs_root)
        except ValueError:
            return f"Error: path escapes the documentation directory: {page_path}"
        if not candidate.is_file():
            available = self.list_doc_pages()[:20]
            listing = "\n".join(f"- {page}" for page in available)
            return (
                f"Error: no documentation page at {page_path}.\n"
                f"Available pages include:\n{listing}"
            )
        return candidate.read_text(encoding="utf-8")

    def list_doc_pages(self) -> List[str]:
        """List generated documentation pages, docs-relative."""
        if not self.docs_root.exists():
            return []
        return sorted(
            str(page.relative_to(self.docs_root))
            for page in self.docs_root.rglob("*.md")
        )
