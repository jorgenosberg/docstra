# File: ./docstra/core/__init__.py

"""
docstra: LLM-powered code documentation assistant.
"""

from collections.abc import Generator
from pathlib import Path

from docstra import __version__ as __version__

from docstra.core.config.settings import (
    ConfigManager,
    ModelProvider,
)
from docstra.core.document_processing.chunking import (
    ChunkingPipeline,
    SemanticChunking,
    SyntaxAwareChunking,
)
from docstra.core.document_processing.document import (
    Document,
)
from docstra.core.document_processing.extractor import (
    DocumentProcessor,
)
from docstra.core.document_processing.parser import CodeParser
from docstra.core.indexing.code_index import (
    CodebaseIndexer,
)
from docstra.core.ingestion.embeddings import (
    EmbeddingFactory,
)
from docstra.core.ingestion.storage import (
    ChromaDBStorage,
    DocumentIndexer,
)
from docstra.core.llm.anthropic import AnthropicClient
from docstra.core.llm.local import LocalModelClient
from docstra.core.llm.ollama import OllamaClient
from docstra.core.llm.openai import OpenAIClient
from docstra.core.retrieval.chroma import ChromaRetriever
from docstra.core.retrieval.hybrid import HybridRetriever


class docstraant:
    """Main entry point for the code documentation assistant."""

    def __init__(self, config_path=None):
        """Initialize the code documentation assistant.

        Args:
            config_path: Optional path to the configuration file
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config

        # Initialize components based on configuration
        self.setup_components()

    def setup_components(self):
        """Set up components based on configuration."""
        # Storage
        storage_dir = self.config.storage.persist_directory
        self.storage = ChromaDBStorage(persist_directory=f"{storage_dir}/chroma")

        # Embedding generator
        self.embedding_generator = EmbeddingFactory.create_embedding_generator(
            embedding_type=self.config.embedding.provider,
            model_name=self.config.embedding.model_name,
            api_key=self.config.embedding.api_key or self.config.model.api_key,
            api_base=self.config.model.api_base,
        )

        # Document processor
        self.document_processor = DocumentProcessor()

        # Code parser
        self.code_parser = CodeParser()

        # Chunking pipeline
        self.chunking_pipeline = ChunkingPipeline(
            [
                SyntaxAwareChunking(),
                SemanticChunking(max_chunk_size=self.config.processing.chunk_size),
            ]
        )

        # Document indexer
        self.document_indexer = DocumentIndexer(
            self.storage,
            self.embedding_generator,
            codebase_root=str(Path.cwd()),
        )

        # Code indexer
        self.code_indexer = CodebaseIndexer(
            index_directory=f"{storage_dir}/index",
            exclude_patterns=self.config.processing.exclude_patterns,
            codebase_root=str(Path.cwd()),
            embedding_backend="chroma",
            embedding_model=self.config.embedding.model_name,
            source_kinds=["tree-sitter"],
        )

        # Retriever
        self.retriever = ChromaRetriever(
            self.storage,
            self.embedding_generator,
            codebase_root=str(Path.cwd()),
        )

        # Hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            self.retriever, self.code_indexer.get_index()
        )

        # LLM client
        self._setup_llm_client()

    def _setup_llm_client(self):
        """Set up the LLM client based on configuration."""
        provider = self.config.model.provider

        if provider == ModelProvider.ANTHROPIC:
            self.llm_client = AnthropicClient(
                model_name=self.config.model.model_name,
                api_key=self.config.model.api_key,
                max_tokens=self.config.model.max_tokens,
                temperature=self.config.model.temperature,
            )
        elif provider == ModelProvider.OPENAI:
            self.llm_client = OpenAIClient(
                model_name=self.config.model.model_name,
                api_key=self.config.model.api_key,
                max_tokens=self.config.model.max_tokens,
                temperature=self.config.model.temperature,
            )
        elif provider == ModelProvider.OLLAMA:
            self.llm_client = OllamaClient(
                model_name=self.config.model.model_name,
                api_base=self.config.model.api_base or "http://localhost:11434",
                max_tokens=self.config.model.max_tokens,
                temperature=self.config.model.temperature,
                validate_connection=False,  # Don't validate during initialization
            )
        elif provider == ModelProvider.LOCAL:
            self.llm_client = LocalModelClient(
                model_name=self.config.model.model_name,
                model_path=self.config.model.model_path,
                max_tokens=self.config.model.max_tokens,
                temperature=self.config.model.temperature,
                device=self.config.model.device,
            )
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    def _require_text_response(self, response: str | Generator[str, None, None]) -> str:
        """Reject streaming responses in the high-level helper interface."""
        if isinstance(response, str):
            return response

        raise TypeError("Streaming responses are not supported in this interface")

    def process_file(self, filepath: str) -> Document:
        """Process a file and prepare it for documentation.

        Args:
            filepath: Path to the file to process

        Returns:
            Processed document
        """
        # Process the file
        document = self.document_processor.process(filepath)

        # Parse the document
        self.code_parser.parse_document(document)

        # Chunk the document
        self.chunking_pipeline.process(document)

        return document

    def index_file(self, filepath: str) -> str:
        """Process and index a file.

        Args:
            filepath: Path to the file to index

        Returns:
            Document ID
        """
        # Process the file
        document = self.process_file(filepath)

        # Index the document
        doc_id = self.document_indexer.index_document(document)
        self.code_indexer.index_document(document)

        return doc_id

    def document_code(
        self, code: str, language: str, additional_context: str = ""
    ) -> str:
        """Generate documentation for code.

        Args:
            code: Code to document
            language: Programming language
            additional_context: Additional context about the code

        Returns:
            Generated documentation
        """
        return self._require_text_response(
            self.llm_client.document_code(
                code=code, language=language, additional_context=additional_context
            )
        )

    def explain_code(
        self, code: str, language: str, additional_context: str = ""
    ) -> str:
        """Generate an explanation for code.

        Args:
            code: Code to explain
            language: Programming language
            additional_context: Additional context about the code

        Returns:
            Generated explanation
        """
        return self._require_text_response(
            self.llm_client.explain_code(
                code=code, language=language, additional_context=additional_context
            )
        )

    def document_file(self, filepath: str) -> str:
        """Generate documentation for a file.

        Args:
            filepath: Path to the file to document

        Returns:
            Generated documentation
        """
        # Process the file
        document = self.process_file(filepath)

        # Generate documentation
        return self.document_code(
            code=document.content,
            language=str(document.metadata.language),
            additional_context=f"File path: {filepath}",
        )

    def explain_file(self, filepath: str) -> str:
        """Generate an explanation for a file.

        Args:
            filepath: Path to the file to explain

        Returns:
            Generated explanation
        """
        # Process the file
        document = self.process_file(filepath)

        # Generate explanation
        return self.explain_code(
            code=document.content,
            language=str(document.metadata.language),
            additional_context=f"File path: {filepath}",
        )

    def answer_question(self, question: str, n_results: int = 5) -> str:
        """Answer a question about the codebase.

        Args:
            question: Question to answer
            n_results: Number of results to retrieve

        Returns:
            Generated answer
        """
        # Retrieve relevant chunks
        results = self.hybrid_retriever.retrieve(
            query=question, n_results=n_results, use_code_context=True
        )

        # Generate answer
        return self._require_text_response(
            self.llm_client.answer_question(question=question, context=results)
        )

    def generate_examples(self, request: str, language: str) -> str:
        """Generate code examples.

        Args:
            request: Request for examples
            language: Programming language

        Returns:
            Generated examples
        """
        return self._require_text_response(
            self.llm_client.generate_examples(request=request, language=language)
        )
