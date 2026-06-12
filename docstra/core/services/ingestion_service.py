# File: ./docstra/core/services/ingestion_service.py
"""
Service responsible for ingesting and indexing codebases.
"""

from pathlib import Path
from typing import List, Optional, Any, Dict
import shutil
import logging

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.panel import Panel
from rich.table import Table

from docstra.core.config.settings import UserConfig
from docstra.core.document_processing.document import Document
from docstra.core.document_processing.extractor import DocumentProcessor
from docstra.core.document_processing.parser import CodeParser
from docstra.core.document_processing.chunking import (
    ChunkingPipeline,
    SemanticChunking,
    SyntaxAwareChunking,
)
from docstra.core.ingestion.embeddings import EmbeddingFactory
from docstra.core.ingestion.fts_storage import FtsStorage
from docstra.core.ingestion.storage import ChromaDBStorage, DocumentIndexer
from docstra.core.indexing.code_index import CodebaseIndex, CodebaseIndexer
from docstra.core.indexing.model import CORE_INDEX_FILENAME
from docstra.core.utils.file_collector import collect_files, FileCollector


class IngestionService:
    """
    Service for ingesting and indexing codebases.
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        callbacks: Optional[List[Any]] = None,
    ):
        """Initialize the ingestion service.

        Args:
            console: Optional console for output
            callbacks: Optional callbacks for tracking
        """
        self.console = console or Console()
        self.callbacks = callbacks
        self.document_processor = DocumentProcessor()
        self.code_parser = CodeParser()

    def ingest_codebase(
        self,
        codebase_path: str,
        user_config: UserConfig,
        force: bool = False,
    ) -> bool:
        """Ingest and index a codebase.

        Args:
            codebase_path: Path to the codebase
            user_config: User configuration
            force: Whether to force reindexing

        Returns:
            True if ingestion was successful, False otherwise
        """
        # Get paths
        codebase_path_abs = Path(codebase_path).resolve()
        persist_directory_name = user_config.storage.persist_directory
        persist_directory = self._resolve_persist_directory(
            codebase_path_abs, persist_directory_name
        )

        # If forcing, remove existing ChromaDB and index directories
        if force:
            chroma_dir = persist_directory / "chroma"
            if chroma_dir.exists() and chroma_dir.is_dir():
                shutil.rmtree(chroma_dir)
            index_dir = persist_directory / "index"
            if index_dir.exists() and index_dir.is_dir():
                shutil.rmtree(index_dir)
            legacy_repo_map = persist_directory / "repo_map.json"
            if legacy_repo_map.exists():
                legacy_repo_map.unlink()
            index_db_path = persist_directory / "index.db"
            if index_db_path.exists():
                index_db_path.unlink()

        index_path = persist_directory / "index"
        core_index_path = index_path / CORE_INDEX_FILENAME
        legacy_index_artifacts = CodebaseIndex.legacy_artifacts_in(index_path)
        legacy_repo_map = persist_directory / "repo_map.json"
        has_legacy_state = bool(legacy_index_artifacts) or legacy_repo_map.exists()

        if has_legacy_state and not force:
            self.console.print(
                "[yellow]Legacy index artifacts detected. Rebuilding the index in the new core manifest format.[/]"
            )
            chroma_dir = persist_directory / "chroma"
            if chroma_dir.exists() and chroma_dir.is_dir():
                shutil.rmtree(chroma_dir)
            if index_path.exists() and index_path.is_dir():
                shutil.rmtree(index_path)
            if legacy_repo_map.exists():
                legacy_repo_map.unlink()

        # Check if already indexed and not forcing
        if core_index_path.exists() and not force:
            self.console.print(
                "[yellow]Codebase already indexed. Use --force to reindex.[/]"
            )
            return True

        # Ensure persistence directory exists
        persist_directory.mkdir(parents=True, exist_ok=True)

        # Get ingestion configuration
        include_dirs = None
        exclude_patterns = None
        if user_config.ingestion:
            include_dirs = user_config.ingestion.include_dirs
            exclude_patterns = (
                user_config.ingestion.exclude_patterns
                or user_config.processing.exclude_patterns
            )
        else:
            exclude_patterns = user_config.processing.exclude_patterns

        # Initialize components
        chunking_pipeline = ChunkingPipeline(
            [
                SyntaxAwareChunking(),
                SemanticChunking(max_chunk_size=user_config.processing.chunk_size),
            ]
        )

        embedding_generator = EmbeddingFactory.create_embedding_generator(
            embedding_type=user_config.embedding.provider,
            model_name=user_config.embedding.model_name,
            api_key=user_config.embedding.api_key or user_config.model.api_key,
            api_base=user_config.model.api_base,
        )

        storage = ChromaDBStorage(persist_directory=str(persist_directory / "chroma"))
        fts_storage = FtsStorage(str(persist_directory / "index.db"))

        doc_indexer = DocumentIndexer(
            storage,
            embedding_generator,
            codebase_root=str(codebase_path_abs),
            fts_storage=fts_storage,
        )

        code_indexer = CodebaseIndexer(
            index_directory=str(persist_directory / "index"),
            exclude_patterns=exclude_patterns or [],
            codebase_root=str(codebase_path_abs),
            embedding_backend="chroma",
            embedding_model=user_config.embedding.model_name,
            source_kinds=["tree-sitter"],
        )

        # Collect files with suppressed logging
        self.console.print(
            f"[cyan]Collecting files from:[/] [bold]{codebase_path_abs}[/]"
        )

        # Temporarily suppress file collector logging to avoid duplication
        file_collector_logger = logging.getLogger("docstra.file_collector")
        original_level = file_collector_logger.level
        file_collector_logger.setLevel(logging.WARNING)

        try:
            file_paths = collect_files(
                base_path=str(codebase_path_abs),
                include_dirs=include_dirs,
                exclude_dirs=exclude_patterns,
                exclude_files=exclude_patterns,
                file_extensions=FileCollector.default_code_file_extensions(),
                log_level=logging.WARNING,  # Suppress INFO logs
            )
        finally:
            # Restore original logging level
            file_collector_logger.setLevel(original_level)

        if not file_paths:
            self.console.print("[yellow]No files found to ingest.[/]")
            return False

        # Show file collection summary
        self._show_collection_summary(file_paths, codebase_path_abs)

        # Show embedding cost estimate if using OpenAI
        if user_config.embedding.provider.lower() == "openai":
            self._show_embedding_cost_estimate(
                file_paths, user_config.embedding.model_name
            )

        # Process, parse, chunk, and index files
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            # Process files
            task_process = progress.add_task(
                "[cyan]Processing files...", total=len(file_paths)
            )

            documents: List[Document] = []
            processing_errors = 0
            for file_path in file_paths:
                try:
                    document = self.document_processor.process(str(file_path))
                    documents.append(document)
                except Exception as e:
                    processing_errors += 1
                    if processing_errors <= 3:  # Only show first few errors
                        self.console.print(
                            f"[yellow]Warning:[/] Failed to process {file_path}: {str(e)}"
                        )
                    elif processing_errors == 4:
                        self.console.print(
                            f"[yellow]Warning:[/] ... and {len(file_paths) - len(documents) - 3} more processing errors"
                        )
                progress.update(task_process, advance=1)

            # Parse documents
            task_parse = progress.add_task(
                "[cyan]Parsing code structure...", total=len(documents)
            )

            parsing_errors = 0
            for document in documents:
                try:
                    self.code_parser.parse_document(document)
                except Exception as e:
                    parsing_errors += 1
                    if parsing_errors <= 3:  # Only show first few errors
                        self.console.print(
                            f"[yellow]Warning:[/] Failed to parse {document.metadata.filepath}: {str(e)}"
                        )
                progress.update(task_parse, advance=1)

            # Chunk documents
            task_chunk = progress.add_task(
                "[cyan]Chunking documents...", total=len(documents)
            )

            chunking_errors = 0
            for document in documents:
                try:
                    chunking_pipeline.process(document)
                except Exception as e:
                    chunking_errors += 1
                    if chunking_errors <= 3:  # Only show first few errors
                        self.console.print(
                            f"[yellow]Warning:[/] Failed to chunk {document.metadata.filepath}: {str(e)}"
                        )
                progress.update(task_chunk, advance=1)

            # Index documents (this is where embeddings are generated)
            task_index = progress.add_task(
                "[cyan]Generating embeddings and indexing...", total=None
            )

            doc_indexer.index_documents(documents)
            code_indexer.index_documents(documents)

            manifest = code_indexer.get_manifest()
            fts_storage.add_symbols(list(manifest.symbols))

            progress.update(
                task_index, completed=True, description="[green]Indexed all documents"
            )

        # Show completion summary with embedding usage
        self._show_completion_summary(
            len(documents),
            processing_errors,
            parsing_errors,
            chunking_errors,
            embedding_generator,
        )

        return True

    def _show_embedding_cost_estimate(
        self, file_paths: List[Path], model_name: str
    ) -> None:
        """Show an estimate of embedding costs for OpenAI models.

        Args:
            file_paths: List of files to be processed
            model_name: OpenAI embedding model name
        """
        # Rough estimate: average 500 tokens per file + chunks
        # This is a conservative estimate since we don't know chunk count yet
        estimated_tokens_per_file = 800  # File content + estimated chunks
        total_estimated_tokens = len(file_paths) * estimated_tokens_per_file

        # Get pricing
        from docstra.core.ingestion.embeddings import EmbeddingUsageTracker

        pricing = EmbeddingUsageTracker.OPENAI_EMBEDDING_PRICING.get(model_name, 0.0001)
        estimated_cost = (total_estimated_tokens / 1000) * pricing

        # Show estimate
        estimate_table = Table(
            title="Embedding Cost Estimate (OpenAI)",
            show_header=True,
            header_style="bold yellow",
        )
        estimate_table.add_column("Metric", style="cyan")
        estimate_table.add_column("Value", justify="right", style="yellow")

        estimate_table.add_row("Model", model_name)
        estimate_table.add_row("Files to process", str(len(file_paths)))
        estimate_table.add_row("Estimated tokens", f"{total_estimated_tokens:,}")
        estimate_table.add_row("Rate per 1K tokens", f"${pricing:.5f}")
        estimate_table.add_row("Estimated cost", f"${estimated_cost:.4f}")

        self.console.print(estimate_table)
        self.console.print(
            "[dim]Note: This is a rough estimate. Actual usage may vary based on file content and chunking.[/]"
        )

    def _show_collection_summary(self, file_paths: List[Path], base_path: Path) -> None:
        """Show a summary of collected files in a nice format."""
        # Count files by directory
        dir_counts: Dict[str, int] = {}
        for file_path in file_paths:
            try:
                rel_dir = str(file_path.parent.relative_to(base_path)) or "."
                dir_counts[rel_dir] = dir_counts.get(rel_dir, 0) + 1
            except ValueError:
                # File not under base_path
                continue

        # Count files by extension
        ext_counts: Dict[str, int] = {}
        for file_path in file_paths:
            ext = file_path.suffix.lower() or "(no extension)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

        # Create summary table
        table = Table(
            title="File Collection Summary", show_header=True, header_style="bold cyan"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")

        table.add_row("Total files found", str(len(file_paths)))

        # Show top directories
        if dir_counts:
            top_dirs = sorted(dir_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            table.add_row("", "")  # Empty row for spacing
            table.add_row("[bold]Top directories:", "")
            for dir_name, count in top_dirs:
                display_name = dir_name if dir_name != "." else "(root)"
                table.add_row(f"  {display_name}", str(count))

        # Show file types
        if ext_counts:
            top_exts = sorted(ext_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            table.add_row("", "")  # Empty row for spacing
            table.add_row("[bold]File types:", "")
            for ext, count in top_exts:
                table.add_row(f"  {ext}", str(count))

        self.console.print(table)

    def _show_completion_summary(
        self,
        successful_docs: int,
        processing_errors: int,
        parsing_errors: int,
        chunking_errors: int,
        embedding_generator: Any,
    ) -> None:
        """Show a completion summary with statistics including embedding usage.

        Args:
            successful_docs: Number of successfully processed documents
            processing_errors: Number of processing errors
            parsing_errors: Number of parsing errors
            chunking_errors: Number of chunking errors
            embedding_generator: The embedding generator used (for usage stats)
        """
        # Create completion panel
        summary_text = (
            f"[bold green]✓ Successfully processed {successful_docs} files[/]\n"
        )

        if processing_errors > 0:
            summary_text += f"[yellow]⚠ {processing_errors} processing errors[/]\n"
        if parsing_errors > 0:
            summary_text += f"[yellow]⚠ {parsing_errors} parsing errors[/]\n"
        if chunking_errors > 0:
            summary_text += f"[yellow]⚠ {chunking_errors} chunking errors[/]\n"

        if processing_errors == 0 and parsing_errors == 0 and chunking_errors == 0:
            summary_text += "[green]No errors encountered during ingestion[/]"

        self.console.print(
            Panel(summary_text, title="[bold green]Ingestion Complete[/]", expand=False)
        )

        # Show embedding usage statistics
        if hasattr(embedding_generator, "get_usage_summary"):
            usage_summary = embedding_generator.get_usage_summary()
            self._show_embedding_usage_summary(usage_summary)

    def _show_embedding_usage_summary(self, usage_summary: Dict[str, Any]) -> None:
        """Show embedding usage summary.

        Args:
            usage_summary: Usage summary from embedding generator
        """
        if not usage_summary or usage_summary.get("total_requests", 0) == 0:
            return

        # Create usage table
        usage_table = Table(
            title="Embedding Usage Summary", show_header=True, header_style="bold blue"
        )
        usage_table.add_column("Metric", style="cyan")
        usage_table.add_column("Value", justify="right", style="green")

        usage_table.add_row(
            "Total tokens processed", f"{usage_summary.get('total_tokens', 0):,}"
        )
        usage_table.add_row(
            "Total API requests", str(usage_summary.get("total_requests", 0))
        )
        usage_table.add_row(
            "Average tokens per request",
            f"{usage_summary.get('average_tokens_per_request', 0):.0f}",
        )

        total_cost = usage_summary.get("total_cost", 0.0)
        if total_cost > 0:
            usage_table.add_row("Total cost", f"${total_cost:.4f}")
            # Add cost breakdown if significant
            if total_cost > 0.01:
                usage_table.add_row("", "")  # Spacing
                usage_table.add_row("[bold]Cost breakdown:", "")
                usage_table.add_row(
                    "  Per 1K tokens",
                    f"${total_cost * 1000 / max(1, usage_summary.get('total_tokens', 1)):.5f}",
                )
        else:
            usage_table.add_row("Total cost", "$0.00 (local model)")

        self.console.print(usage_table)

    def _resolve_persist_directory(
        self, codebase_path: Path, persist_directory_name: str
    ) -> Path:
        """Resolve the persistence directory path.

        Args:
            codebase_path: Path to the codebase
            persist_directory_name: Name of the persistence directory

        Returns:
            Resolved persistence directory path
        """
        persist_directory = Path(persist_directory_name)

        # If the path is relative, resolve it relative to the codebase path
        if not persist_directory.is_absolute():
            persist_directory = codebase_path / persist_directory

        return persist_directory.resolve()
