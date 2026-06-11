# File: ./docstra/core/retrieval/chroma.py

"""
Document retrieval using ChromaDB.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from docstra.core.ingestion.embeddings import EmbeddingGenerator
from docstra.core.ingestion.storage import ChromaDBStorage
from docstra.core.indexing.model import normalize_file_id


class ChromaRetriever:
    """Retriever for documents and chunks using ChromaDB."""

    def __init__(
        self,
        storage: ChromaDBStorage,
        embedding_generator: EmbeddingGenerator,
        codebase_root: Optional[str] = None,
    ):
        """Initialize the ChromaDB retriever.

        Args:
            storage: ChromaDB storage
            embedding_generator: Generator for creating embeddings
        """
        self.storage = storage
        self.embedding_generator = embedding_generator
        self.codebase_root = codebase_root

    def retrieve_documents(
        self, query: str, n_results: int = 10, **filters
    ) -> List[Dict[str, Any]]:
        """Retrieve documents by similarity to a query.

        Args:
            query: Query string
            n_results: Number of results to return
            **filters: Additional filters to apply

        Returns:
            List of matching documents
        """
        # Generate embedding for the query
        query_embedding = self.embedding_generator.generate_embedding(query)

        # Search for similar documents
        results = self.storage.search_documents(
            query_embedding=query_embedding, n_results=n_results, **filters
        )

        return results

    def retrieve_chunks(
        self, query: str, n_results: int = 20, **filters
    ) -> List[Dict[str, Any]]:
        """Retrieve document chunks by similarity to a query.

        Args:
            query: Query string
            n_results: Number of results to return
            **filters: Additional filters to apply

        Returns:
            List of matching chunks
        """
        # Generate embedding for the query
        query_embedding = self.embedding_generator.generate_embedding(query)

        # Search for similar chunks
        results = self.storage.search_chunks(
            query_embedding=query_embedding, n_results=n_results, **filters
        )

        return results

    def retrieve_by_context(
        self, query: str, context_type: str, context_value: str, n_results: int = 20
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks filtered by a specific context.

        Args:
            query: Query string
            context_type: Type of context to filter by (e.g., "language", "document_id")
            context_value: Value to filter on
            n_results: Number of results to return

        Returns:
            List of matching chunks
        """
        # Apply context as a filter
        filters = {context_type: context_value}

        return self.retrieve_chunks(query=query, n_results=n_results, **filters)

    def retrieve_by_filepath(
        self, query: str, filepath: str, n_results: int = 20
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks from a specific file.

        Args:
            query: Query string
            filepath: Path to the file
            n_results: Number of results to return

        Returns:
            List of matching chunks
        """
        file_id = normalize_file_id(filepath, self.codebase_root)
        return self.retrieve_by_context(
            query=query,
            context_type="document_id",
            context_value=file_id,
            n_results=n_results,
        )

    def retrieve_by_language(
        self, query: str, language: str, n_results: int = 20
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks in a specific programming language.

        Args:
            query: Query string
            language: Programming language
            n_results: Number of results to return

        Returns:
            List of matching chunks
        """
        return self.retrieve_by_context(
            query=query,
            context_type="language",
            context_value=language,
            n_results=n_results,
        )

    def get_context_for_document(self, document_id: str) -> Dict[str, Any]:
        """Get the full context for a document.

        Args:
            document_id: Document ID

        Returns:
            Document and its chunks
        """
        normalized_id = normalize_file_id(document_id, self.codebase_root)
        document = self.storage.get_document(normalized_id)
        chunks = self.storage.get_chunks_for_document(normalized_id)

        return {"document": document, "chunks": chunks}

    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID.

        Args:
            document_id: Document ID

        Returns:
            The document if found, None otherwise
        """
        normalized_id = normalize_file_id(document_id, self.codebase_root)
        return self.storage.get_document(normalized_id)

    def get_chunks_for_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document.

        Args:
            document_id: Document ID

        Returns:
            List of chunks for the document
        """
        normalized_id = normalize_file_id(document_id, self.codebase_root)
        return self.storage.get_chunks_for_document(normalized_id)
