# File: ./docstra/core/services/documentation_service.py
"""
Service responsible for generating documentation for the codebase.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from docstra.core.config.settings import DocumentationConfig, ModelProvider, UserConfig
from docstra.core.document_processing.document import Document
from docstra.core.document_processing.extractor import DocumentProcessor

# Old generator import removed - now using EnhancedDocumentationGenerator
from docstra.core.indexing.code_index import (
    CodebaseIndexer,
)  # For loading index for repo_map
from docstra.core.indexing.code_index import CodebaseIndex
from docstra.core.indexing.model import CORE_INDEX_FILENAME
from docstra.core.indexing.repo_map import RepositoryMap
from docstra.core.ingestion.embeddings import EmbeddingFactory
from docstra.core.ingestion.storage import ChromaDBStorage
from docstra.core.llm.base import LLMClient
from docstra.core.retrieval.chroma import ChromaRetriever

# Using the same LLM client factory as ChatService for consistency
# This assumes _get_llm_client_for_chat_service is suitable or will be generalized
from docstra.core.services.chat_service import (
    _get_llm_client_for_chat_service as _get_llm_client_for_doc_service,
)

# Import incremental documentation components
from docstra.core.documentation.dependencies import DocumentationDependencyTracker
from docstra.core.documentation.overrides import DocumentationOverrideManager
from docstra.core.services.change_detection_service import ChangeDetectionService
from docstra.core.utils.file_collector import (
    FileCollector,
    collect_files,
    filter_files_with_patterns,
)


class DocumentationService:
    """
    Handles the generation of codebase documentation.
    """

    def __init__(
        self,
        user_config: UserConfig,
        console: Optional[Console] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> None:
        self.user_config = user_config
        self.doc_config: DocumentationConfig
        # Handle the case where documentation might be None
        if user_config.documentation is None:
            # Create a default documentation config
            self.doc_config = DocumentationConfig()
        else:
            self.doc_config = user_config.documentation
        self.console = console or Console()
        self.callbacks = callbacks

        self.llm_client: LLMClient = _get_llm_client_for_doc_service(
            self.user_config, self.callbacks
        )
        self.document_processor = DocumentProcessor()

        # Initialize incremental documentation components
        self.dependency_tracker: Optional[DocumentationDependencyTracker] = None
        self.override_manager: Optional[DocumentationOverrideManager] = None
        self.change_detector: Optional[ChangeDetectionService] = None

    def generate_documentation(
        self,
        input_path_str: str,
        output_dir_str: Optional[str] = None,
        doc_format_str: Optional[str] = None,
        project_name_str: Optional[str] = None,
        project_description_str: Optional[str] = None,
        theme_str: Optional[str] = None,
        structure_str: Optional[str] = None,
        module_depth_str: Optional[str] = None,
        llm_style_prompt_str: Optional[str] = None,
        cli_include_patterns: Optional[List[str]] = None,
        cli_exclude_patterns: Optional[List[str]] = None,
        max_workers_override: Optional[int] = None,
        incremental: bool = False,
    ) -> bool:
        self.console.print(
            Panel("Starting Documentation Generation", style="bold blue")
        )

        effective_output_dir = Path(output_dir_str or self.doc_config.output_dir)
        effective_format = doc_format_str or self.doc_config.format
        input_path_abs = Path(input_path_str).resolve()
        effective_project_name = (
            project_name_str or self.doc_config.project_name or input_path_abs.name
        )
        effective_project_description = (
            project_description_str or self.doc_config.project_description or ""
        )
        effective_llm_style_prompt = (
            llm_style_prompt_str or self.doc_config.llm_style_prompt
        )

        effective_include_dirs = (
            cli_include_patterns
            if cli_include_patterns is not None
            else (self.doc_config.include_dirs or [])
        )
        effective_exclude_patterns = (
            cli_exclude_patterns
            if cli_exclude_patterns is not None
            else (self.doc_config.exclude_patterns or [])
        )

        if max_workers_override is not None:
            effective_max_workers = max(1, max_workers_override)
        else:
            provider_specific_workers = None
            if (
                self.user_config.model.provider == ModelProvider.OLLAMA
                and self.doc_config.max_workers_ollama is not None
            ):
                provider_specific_workers = self.doc_config.max_workers_ollama
            elif (
                self.user_config.model.provider
                in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC]
                and self.doc_config.max_workers_api is not None
            ):
                provider_specific_workers = self.doc_config.max_workers_api

            if provider_specific_workers is not None:
                effective_max_workers = provider_specific_workers
            elif self.doc_config.max_workers_default is not None:
                effective_max_workers = self.doc_config.max_workers_default
            else:
                effective_max_workers = os.cpu_count() or 2
            effective_max_workers = max(1, effective_max_workers)

        self.console.print(f"  Input path: [cyan]{input_path_abs}[/cyan]")
        self.console.print(f"  Output directory: [cyan]{effective_output_dir}[/cyan]")
        self.console.print(f"  Format: [cyan]{effective_format}[/cyan]")
        self.console.print(f"  Project Name: [cyan]{effective_project_name}[/cyan]")
        self.console.print(f"  Max Workers: [cyan]{effective_max_workers}[/cyan]")
        self.console.print(f"  Incremental: [cyan]{incremental}[/cyan]")

        effective_output_dir.mkdir(parents=True, exist_ok=True)

        persist_dir_name = self.user_config.storage.persist_directory
        base_persist_path = Path(persist_dir_name)
        if not base_persist_path.is_absolute():
            base_persist_path = input_path_abs / persist_dir_name
        abs_persist_directory = base_persist_path.resolve()

        # Initialize incremental documentation components
        self.dependency_tracker = DocumentationDependencyTracker(
            str(abs_persist_directory)
        )
        self.override_manager = DocumentationOverrideManager(str(abs_persist_directory))
        self.change_detector = ChangeDetectionService(str(abs_persist_directory))

        self.console.print(
            f"[dim]Collecting files from {input_path_abs}, persist_dir for ignores: {abs_persist_directory}[/dim]"
        )
        all_files_for_gen = collect_files(
            base_path=str(input_path_abs),
            file_extensions=FileCollector.default_code_file_extensions(),
        )
        self.console.print(
            f"Collected {len(all_files_for_gen)} files (before filtering)."
        )

        docs_target_file_paths = filter_files_with_patterns(
            file_paths=all_files_for_gen,
            base_path=str(input_path_abs),
            include_dirs=effective_include_dirs,
            exclude_patterns=effective_exclude_patterns,
        )
        self.console.print(
            f"Filtered down to {len(docs_target_file_paths)} files for documentation."
        )

        if not docs_target_file_paths:
            self.console.print(
                "[yellow]No files to document after filtering. Exiting.[/yellow]"
            )
            return True

        documents_for_generation: List[Document] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            task_load = progress.add_task(
                "[cyan]Loading files...", total=len(docs_target_file_paths)
            )
            for file_path in docs_target_file_paths:
                try:
                    document = self.document_processor.process(str(file_path))
                    if document:
                        document.metadata.filepath = str(
                            file_path.relative_to(input_path_abs)
                        )
                        documents_for_generation.append(document)
                except Exception as e:
                    self.console.print(
                        f"[yellow]Failed to process {file_path}: {e}[/yellow]"
                    )
                progress.update(task_load, advance=1)

        if not documents_for_generation:
            self.console.print(
                "[bold red]No documents could be successfully loaded for generation. Exiting.[/bold red]"
            )
            return False

        repo_map: Optional[RepositoryMap] = None
        chroma_retriever: Optional[ChromaRetriever] = None
        code_indexer_path = abs_persist_directory / "index"
        core_index_path = code_indexer_path / CORE_INDEX_FILENAME
        chroma_storage_path = abs_persist_directory / "chroma"
        legacy_index_artifacts = CodebaseIndex.legacy_artifacts_in(code_indexer_path)
        legacy_repo_map = abs_persist_directory / "repo_map.json"

        if not core_index_path.exists() and (
            legacy_index_artifacts or legacy_repo_map.exists()
        ):
            raise ValueError(
                "Legacy Docstra index artifacts were found. Run 'docstra ingest' "
                "to rebuild the index in the new format."
            )

        if core_index_path.exists():
            try:
                temp_code_index = CodebaseIndexer(
                    index_directory=str(code_indexer_path),
                    codebase_root=str(input_path_abs),
                ).get_index()
                if temp_code_index:
                    repo_map = RepositoryMap(str(input_path_abs), temp_code_index)
                    repo_map.build()
                    self.console.print("[dim]Repo map created from core index.[/dim]")
            except Exception as e_map:
                self.console.print(
                    f"[yellow]Warning: Could not load repository map: {e_map}[/yellow]"
                )

        if chroma_storage_path.exists():
            try:
                embedding_gen = EmbeddingFactory.create_embedding_generator(
                    embedding_type=self.user_config.embedding.provider,
                    model_name=self.user_config.embedding.model_name,
                    api_key=self.user_config.embedding.api_key
                    or self.user_config.model.api_key,
                    api_base=self.user_config.model.api_base,
                )
                chroma_db = ChromaDBStorage(str(chroma_storage_path))
                chroma_retriever = ChromaRetriever(
                    chroma_db,
                    embedding_gen,
                    codebase_root=str(input_path_abs),
                )
                self.console.print(
                    f"[dim]ChromaRetriever initialized from {chroma_storage_path}.[/dim]"
                )
            except Exception as e_chroma:
                self.console.print(
                    f"[yellow]Warning: Could not initialize ChromaRetriever: {e_chroma}[/yellow]"
                )

        # Get code index if available
        code_index = None
        if core_index_path.exists():
            try:
                indexer = CodebaseIndexer(
                    index_directory=str(abs_persist_directory / "index"),
                    codebase_root=str(input_path_abs),
                )
                code_index = indexer.get_index()
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning: Could not load code index: {e}[/yellow]"
                )

        # Import the enhanced generator
        from docstra.core.documentation.generator import DocumentationGenerator

        doc_generator = DocumentationGenerator(
            llm_client=self.llm_client,
            output_dir=effective_output_dir,
            repo_map=repo_map,
            chroma_retriever=chroma_retriever,
            code_index=code_index,
            project_name=effective_project_name,
            project_description=effective_project_description,
            console=self.console,
            max_workers=effective_max_workers,
            documentation_depth="comprehensive",
            style_guide=effective_llm_style_prompt,
        )

        self.console.print(
            f"Generating documentation with {effective_max_workers} worker(s)..."
        )
        try:
            success = doc_generator.generate_documentation(
                documents=documents_for_generation,
                generate_guides=True,
                generate_api_docs=True,
                generate_cross_references=True,
            )

            if success:
                self.console.print(
                    "[bold green]Documentation generation process finished.[/bold green]"
                )
                self.console.print(
                    f"Output at: [link=file://{effective_output_dir.resolve()}]{effective_output_dir.resolve()}[/link]"
                )
                return True
            else:
                self.console.print(
                    "[bold red]Documentation generation failed.[/bold red]"
                )
                return False
        except Exception as e_gen:
            self.console.print(
                f"[bold red]Error during documentation generation: {e_gen}[/bold red]"
            )
            return False

    def generate_incremental_documentation(
        self,
        input_path_str: str,
        output_dir_str: Optional[str] = None,
        base_ref: str = "HEAD~1",
        force_files: Optional[List[str]] = None,
        **kwargs,
    ) -> bool:
        """Generate documentation incrementally based on detected changes.

        Args:
            input_path_str: Path to the codebase
            output_dir_str: Output directory for documentation
            base_ref: Git reference to compare against for changes
            force_files: List of files to force regeneration
            **kwargs: Additional arguments passed to generate_documentation

        Returns:
            True if successful, False otherwise
        """
        input_path_abs = Path(input_path_str).resolve()

        # Setup persist directory
        persist_dir_name = self.user_config.storage.persist_directory
        base_persist_path = Path(persist_dir_name)
        if not base_persist_path.is_absolute():
            base_persist_path = input_path_abs / persist_dir_name
        abs_persist_directory = base_persist_path.resolve()

        # Initialize services
        dependency_tracker = DocumentationDependencyTracker(str(abs_persist_directory))
        override_manager = DocumentationOverrideManager(str(abs_persist_directory))
        change_detector = ChangeDetectionService(str(abs_persist_directory))

        self.console.print(
            Panel("Incremental Documentation Generation", style="bold green")
        )

        # Detect changes
        if force_files:
            change_analysis = change_detector.detect_changes_from_file_list(force_files)
            self.console.print(
                f"[yellow]Forced regeneration for {len(force_files)} files[/yellow]"
            )
        else:
            change_analysis = change_detector.detect_changes_from_git(
                str(input_path_abs), base_ref
            )
            self.console.print(
                f"[dim]Detected changes using Git (base: {base_ref})[/dim]"
            )

        if not change_analysis.has_changes:
            self.console.print(
                "[green]No changes detected. Documentation is up to date.[/green]"
            )
            return True

        self.console.print("[cyan]Changes detected:[/cyan]")
        self.console.print(f"  • Changed files: {len(change_analysis.changed_files)}")
        self.console.print(f"  • New files: {len(change_analysis.new_files)}")
        self.console.print(f"  • Deleted files: {len(change_analysis.deleted_files)}")

        # Get impacted documentation
        all_changed_files = change_analysis.changed_files + change_analysis.new_files
        impacted_docs = dependency_tracker.get_impacted_documentation(all_changed_files)

        self.console.print(
            f"[yellow]Impacted documentation pages: {len(impacted_docs)}[/yellow]"
        )

        # Filter files through overrides
        files_to_process = []
        skipped_by_override = []

        for file_path in all_changed_files:
            if override_manager.should_skip_generation(file_path):
                skipped_by_override.append(file_path)
            else:
                files_to_process.append(file_path)

        if skipped_by_override:
            self.console.print(
                f"[dim]Skipped {len(skipped_by_override)} files due to overrides[/dim]"
            )

        if not files_to_process:
            self.console.print(
                "[yellow]All changed files are skipped by overrides. Nothing to process.[/yellow]"
            )
            return True

        # Generate documentation for changed files only
        self.console.print(f"[cyan]Processing {len(files_to_process)} files...[/cyan]")

        # Prepare kwargs for incremental generation
        incremental_kwargs = kwargs.copy()
        incremental_kwargs["incremental"] = True

        # Process files in smaller batches for incremental updates
        success = self.generate_documentation(
            input_path_str=input_path_str,
            output_dir_str=output_dir_str,
            **incremental_kwargs,
        )

        if success:
            # Mark generation as complete
            change_detector.mark_generation_complete(str(input_path_abs))
            self.console.print(
                "[bold green]Incremental documentation generation completed successfully![/bold green]"
            )
        else:
            self.console.print(
                "[bold red]Incremental documentation generation failed.[/bold red]"
            )

        return success

    def get_documentation_status(self, input_path_str: str) -> Dict[str, Any]:
        """Get status of documentation including changes and dependencies.

        Args:
            input_path_str: Path to the codebase

        Returns:
            Dictionary containing documentation status information
        """
        input_path_abs = Path(input_path_str).resolve()

        # Setup persist directory
        persist_dir_name = self.user_config.storage.persist_directory
        base_persist_path = Path(persist_dir_name)
        if not base_persist_path.is_absolute():
            base_persist_path = input_path_abs / persist_dir_name
        abs_persist_directory = base_persist_path.resolve()

        # Initialize services
        dependency_tracker = DocumentationDependencyTracker(str(abs_persist_directory))
        override_manager = DocumentationOverrideManager(str(abs_persist_directory))
        change_detector = ChangeDetectionService(str(abs_persist_directory))

        # Get status information
        status = {
            "codebase_path": str(input_path_abs),
            "last_generation": change_detector.get_last_generation_info(),
            "dependency_stats": dependency_tracker.get_dependency_stats(),
            "override_stats": override_manager.get_override_stats(),
            "outdated_docs": dependency_tracker.get_outdated_documentation(),
            "change_history": change_detector.get_change_history(5),
        }

        # Detect current changes
        change_analysis = change_detector.detect_changes_since_last_generation(
            str(input_path_abs)
        )
        status["current_changes"] = {
            "has_changes": change_analysis.has_changes,
            "total_changes": change_analysis.total_changes,
            "changed_files": len(change_analysis.changed_files),
            "new_files": len(change_analysis.new_files),
            "deleted_files": len(change_analysis.deleted_files),
        }

        return status
