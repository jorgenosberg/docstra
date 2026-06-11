# File: ./docstra/core/ingestion/embeddings.py

"""
Vector embedding generation for code documents.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import requests
import tiktoken

from docstra.core.document_processing.document import Document
from docstra.core.indexing.model import make_chunk_id, normalize_file_id


def _vector_to_list(vector: Sequence[float]) -> List[float]:
    """Convert any embedding-like sequence into a plain float list."""
    return [float(value) for value in vector]


class EmbeddingUsageTracker:
    """Tracks token usage and costs for embedding generation."""

    # OpenAI embedding pricing per 1K tokens (as of 2024)
    OPENAI_EMBEDDING_PRICING = {
        "text-embedding-3-small": 0.00002,
        "text-embedding-3-large": 0.00013,
        "text-embedding-ada-002": 0.0001,
    }

    def __init__(self) -> None:
        """Initialize the usage tracker."""
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_requests = 0
        self.usage_history: List[Dict[str, Any]] = []

    def _estimate_tokens(self, text: str, model: str = "text-embedding-3-small") -> int:
        """Estimate token count for text using tiktoken."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            return len(text) // 4

    def record_usage(
        self,
        provider: str,
        model: str,
        texts: List[str],
        request_type: str = "embedding",
    ) -> Dict[str, Any]:
        """Record embedding usage."""
        total_tokens = sum(self._estimate_tokens(text, model) for text in texts)

        cost = 0.0
        if provider.lower() == "openai":
            rate = self.OPENAI_EMBEDDING_PRICING.get(model, 0.0001)
            cost = (total_tokens / 1000) * rate

        self.total_tokens += total_tokens
        self.total_cost += cost
        self.total_requests += 1

        usage_record = {
            "provider": provider,
            "model": model,
            "tokens": total_tokens,
            "cost": cost,
            "num_texts": len(texts),
            "request_type": request_type,
            "timestamp": time.time(),
        }

        self.usage_history.append(usage_record)
        return usage_record

    def get_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
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
        """Generate an embedding for a single text."""

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary for this generator."""
        return self.usage_tracker.get_summary()


class HuggingFaceEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator using sentence-transformers directly."""

    def __init__(
        self, model_name: str = "sentence-transformers/all-mpnet-base-v2"
    ) -> None:
        super().__init__()
        self.model_name = model_name

        from sentence_transformers import SentenceTransformer

        try:
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        except TypeError:
            self.model = SentenceTransformer(model_name)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return [_vector_to_list(vector) for vector in embeddings]

    def generate_embedding(self, text: str) -> List[float]:
        self.usage_tracker.record_usage("huggingface", self.model_name, [text])
        return self._encode([text])[0]

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        self.usage_tracker.record_usage("huggingface", self.model_name, texts)
        return self._encode(texts)


class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator using the OpenAI SDK directly."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY or embedding.api_key."
            )

        from openai import OpenAI

        self.client = OpenAI(api_key=resolved_api_key, base_url=api_base)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [_vector_to_list(item.embedding) for item in response.data]

    def generate_embedding(self, text: str) -> List[float]:
        self.usage_tracker.record_usage("openai", self.model_name, [text])
        return self._embed([text])[0]

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        self.usage_tracker.record_usage("openai", self.model_name, texts)
        return self._embed(texts)


class OllamaEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator using Ollama's HTTP embedding endpoints."""

    def __init__(
        self,
        model_name: str = "llama3.2",
        api_base: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.api_base = (api_base or os.environ.get("OLLAMA_API_BASE") or "").rstrip(
            "/"
        ) or "http://localhost:11434"
        self.timeout = timeout

    def _embed_with_current_endpoint(self, texts: List[str]) -> List[List[float]]:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "input": texts if len(texts) > 1 else texts[0],
        }
        response = requests.post(
            f"{self.api_base}/api/embed",
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code == 404:
            raise FileNotFoundError("Ollama /api/embed endpoint is unavailable")
        response.raise_for_status()
        data = response.json()
        if "embeddings" not in data:
            raise ValueError("Ollama response did not include embeddings")
        return [_vector_to_list(vector) for vector in data["embeddings"]]

    def _embed_with_legacy_endpoint(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            response = requests.post(
                f"{self.api_base}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            if "embedding" not in data:
                raise ValueError("Legacy Ollama response did not include an embedding")
            embeddings.append(_vector_to_list(data["embedding"]))
        return embeddings

    def _embed(self, texts: List[str]) -> List[List[float]]:
        try:
            return self._embed_with_current_endpoint(texts)
        except FileNotFoundError:
            return self._embed_with_legacy_endpoint(texts)

    def generate_embedding(self, text: str) -> List[float]:
        self.usage_tracker.record_usage("ollama", self.model_name, [text])
        return self._embed([text])[0]

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        self.usage_tracker.record_usage("ollama", self.model_name, texts)
        return self._embed(texts)


class EmbeddingFactory:
    """Factory for creating embedding generators."""

    @staticmethod
    def create_embedding_generator(embedding_type: str, **kwargs) -> EmbeddingGenerator:
        """Create an embedding generator based on type."""
        if embedding_type.lower() == "huggingface":
            model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
            return HuggingFaceEmbeddingGenerator(model_name=model_name)
        if embedding_type.lower() == "openai":
            model_name = kwargs.get("model_name", "text-embedding-3-small")
            return OpenAIEmbeddingGenerator(
                model_name=model_name,
                api_key=kwargs.get("api_key"),
                api_base=kwargs.get("api_base"),
            )
        if embedding_type.lower() == "ollama":
            model_name = kwargs.get("model_name", "llama3.2")
            return OllamaEmbeddingGenerator(
                model_name=model_name,
                api_base=kwargs.get("api_base"),
            )
        raise ValueError(f"Unsupported embedding type: {embedding_type}")


class DocumentEmbedder:
    """Generate embeddings for documents and their chunks."""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        codebase_root: Optional[str] = None,
    ) -> None:
        """Initialize the document embedder."""
        self.embedding_generator = embedding_generator
        self.codebase_root = codebase_root

    def embed_document(self, document: Document) -> Dict[str, List[float]]:
        """Generate embeddings for a document and its chunks."""
        doc_id = normalize_file_id(document.metadata.filepath, self.codebase_root)
        doc_embedding = self.embedding_generator.generate_embedding(document.content)
        chunk_embeddings: Dict[str, List[float]] = {}

        if document.chunks:
            chunk_texts = [chunk.content for chunk in document.chunks]
            chunk_embedding_vectors = self.embedding_generator.generate_embeddings(
                chunk_texts
            )

            for i, chunk in enumerate(document.chunks):
                chunk_id = make_chunk_id(doc_id, chunk.start_line, chunk.end_line)
                chunk_embeddings[chunk_id] = chunk_embedding_vectors[i]

        chunk_embeddings[doc_id] = doc_embedding

        return chunk_embeddings

    def embed_documents(
        self, documents: List[Document]
    ) -> Dict[str, Dict[str, List[float]]]:
        """Generate embeddings for multiple documents and their chunks."""
        embeddings: Dict[str, Dict[str, List[float]]] = {}

        for document in documents:
            doc_id = normalize_file_id(document.metadata.filepath, self.codebase_root)
            embeddings[doc_id] = self.embed_document(document)

        return embeddings
