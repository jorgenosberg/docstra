# File: ./docstra/core/ingestion/embeddings.py

"""
Vector embedding generation for code documents.
"""

from __future__ import annotations

import os
import tiktoken
from abc import ABC, abstractmethod
from typing import Dict, List, Any

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings

from docstra.core.document_processing.document import Document


class EmbeddingUsageTracker:
    """Tracks token usage and costs for embedding generation."""

    # OpenAI embedding pricing per 1K tokens (as of 2024)
    OPENAI_EMBEDDING_PRICING = {
        "text-embedding-3-small": 0.00002,  # $0.00002 per 1K tokens
        "text-embedding-3-large": 0.00013,  # $0.00013 per 1K tokens
        "text-embedding-ada-002": 0.0001,  # $0.0001 per 1K tokens
    }

    def __init__(self) -> None:
        """Initialize the usage tracker."""
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_requests = 0
        self.usage_history: List[Dict[str, Any]] = []

    def _estimate_tokens(self, text: str, model: str = "text-embedding-3-small") -> int:
        """Estimate token count for text using tiktoken.

        Args:
            text: Text to estimate tokens for
            model: Model name for encoding selection

        Returns:
            Estimated token count
        """
        try:
            # Use cl100k_base encoding for OpenAI embedding models
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback: rough estimate of 4 characters per token
            return len(text) // 4

    def record_usage(
        self,
        provider: str,
        model: str,
        texts: List[str],
        request_type: str = "embedding",
    ) -> Dict[str, Any]:
        """Record embedding usage.

        Args:
            provider: Embedding provider (openai, huggingface, ollama)
            model: Model name
            texts: List of texts that were embedded
            request_type: Type of request

        Returns:
            Usage information dictionary
        """
        # Calculate token usage
        total_tokens = sum(self._estimate_tokens(text, model) for text in texts)

        # Calculate cost (only for OpenAI)
        cost = 0.0
        if provider.lower() == "openai":
            rate = self.OPENAI_EMBEDDING_PRICING.get(model, 0.0001)  # Default rate
            cost = (total_tokens / 1000) * rate

        # Update totals
        self.total_tokens += total_tokens
        self.total_cost += cost
        self.total_requests += 1

        # Create usage record
        usage_record = {
            "provider": provider,
            "model": model,
            "tokens": total_tokens,
            "cost": cost,
            "num_texts": len(texts),
            "request_type": request_type,
            "timestamp": __import__("time").time(),
        }

        self.usage_history.append(usage_record)
        return usage_record

    def get_summary(self) -> Dict[str, Any]:
        """Get usage summary.

        Returns:
            Summary of usage statistics
        """
        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "total_requests": self.total_requests,
            "average_tokens_per_request": self.total_tokens
            / max(1, self.total_requests),
        }


class EmbeddingGenerator(ABC):
    """Abstract base class for embedding generators."""

    def __init__(self) -> None:
        """Initialize the embedding generator."""
        self.usage_tracker = EmbeddingUsageTracker()

    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to generate an embedding for

        Returns:
            The embedding vector
        """
        pass

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: The texts to generate embeddings for

        Returns:
            List of embedding vectors
        """
        pass

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary for this generator.

        Returns:
            Usage summary dictionary
        """
        return self.usage_tracker.get_summary()


class HuggingFaceEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator using HuggingFace models."""

    def __init__(
        self, model_name: str = "sentence-transformers/all-mpnet-base-v2"
    ) -> None:
        """Initialize the HuggingFace embedding generator.

        Args:
            model_name: Name of the HuggingFace model to use
        """
        super().__init__()
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs={"trust_remote_code": True}
        )

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to generate an embedding for

        Returns:
            The embedding vector
        """
        # Track usage (no cost for HuggingFace local models)
        self.usage_tracker.record_usage("huggingface", self.model_name, [text])
        return self.embeddings.embed_query(text)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: The texts to generate embeddings for

        Returns:
            List of embedding vectors
        """
        # Track usage (no cost for HuggingFace local models)
        self.usage_tracker.record_usage("huggingface", self.model_name, texts)
        return self.embeddings.embed_documents(texts)


class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator using OpenAI embedding models."""

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        """Initialize the OpenAI embedding generator.

        Args:
            model_name: Name of the OpenAI embedding model to use
        """
        super().__init__()
        self.model_name = model_name
        # Get API key from environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )

        self.embeddings = OpenAIEmbeddings(model=model_name)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to generate an embedding for

        Returns:
            The embedding vector
        """
        # Track usage and cost
        self.usage_tracker.record_usage("openai", self.model_name, [text])
        return self.embeddings.embed_query(text)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: The texts to generate embeddings for

        Returns:
            List of embedding vectors
        """
        # Track usage and cost
        self.usage_tracker.record_usage("openai", self.model_name, texts)
        return self.embeddings.embed_documents(texts)


class OllamaEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator using Ollama models."""

    def __init__(self, model_name: str = "llama3.2") -> None:
        """Initialize the Ollama embedding generator.

        Args:
            model_name: Name of the Ollama model to use
        """
        super().__init__()
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(model=model_name)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to generate an embedding for

        Returns:
            The embedding vector
        """
        # Track usage (no cost for local Ollama models)
        self.usage_tracker.record_usage("ollama", self.model_name, [text])
        return self.embeddings.embed_query(text)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: The texts to generate embeddings for

        Returns:
            List of embedding vectors
        """
        # Track usage (no cost for local Ollama models)
        self.usage_tracker.record_usage("ollama", self.model_name, texts)
        return self.embeddings.embed_documents(texts)


class EmbeddingFactory:
    """Factory for creating embedding generators."""

    @staticmethod
    def create_embedding_generator(embedding_type: str, **kwargs) -> EmbeddingGenerator:
        """Create an embedding generator based on type.

        Args:
            embedding_type: Type of embedding generator to create
            **kwargs: Additional arguments for the generator

        Returns:
            An embedding generator

        Raises:
            ValueError: If the embedding type is not supported
        """
        if embedding_type.lower() == "huggingface":
            model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
            return HuggingFaceEmbeddingGenerator(model_name=model_name)
        elif embedding_type.lower() == "openai":
            model_name = kwargs.get("model_name", "text-embedding-3-small")
            return OpenAIEmbeddingGenerator(model_name=model_name)
        elif embedding_type.lower() == "ollama":
            model_name = kwargs.get("model_name", "llama3.2")
            return OllamaEmbeddingGenerator(model_name=model_name)
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")


class DocumentEmbedder:
    """Generate embeddings for documents and their chunks."""

    def __init__(self, embedding_generator: EmbeddingGenerator) -> None:
        """Initialize the document embedder.

        Args:
            embedding_generator: Generator for creating embeddings
        """
        self.embedding_generator = embedding_generator

    def embed_document(self, document: Document) -> Dict[str, List[float]]:
        """Generate embeddings for a document and its chunks.

        Args:
            document: The document to embed

        Returns:
            Dictionary mapping chunk IDs to embeddings
        """
        # Generate a document-level embedding
        doc_embedding = self.embedding_generator.generate_embedding(document.content)

        # Generate embeddings for each chunk
        chunk_embeddings: Dict[str, List[float]] = {}

        if document.chunks:
            chunk_texts = [chunk.content for chunk in document.chunks]
            chunk_embedding_vectors = self.embedding_generator.generate_embeddings(
                chunk_texts
            )

            for i, chunk in enumerate(document.chunks):
                chunk_id = (
                    f"{document.metadata.filepath}#{chunk.start_line}-{chunk.end_line}"
                )
                chunk_embeddings[chunk_id] = chunk_embedding_vectors[i]

        # Include the document-level embedding
        doc_id = document.metadata.filepath
        chunk_embeddings[doc_id] = doc_embedding

        return chunk_embeddings

    def embed_documents(
        self, documents: List[Document]
    ) -> Dict[str, Dict[str, List[float]]]:
        """Generate embeddings for multiple documents and their chunks.

        Args:
            documents: The documents to embed

        Returns:
            Dictionary mapping document IDs to chunk embeddings
        """
        embeddings: Dict[str, Dict[str, List[float]]] = {}

        for document in documents:
            doc_id = document.metadata.filepath
            embeddings[doc_id] = self.embed_document(document)

        return embeddings
