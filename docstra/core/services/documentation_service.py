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
from docstra.core.documentation.pipeline import (
    compute_impacted_file_ids,
    doc_relative_path,
    file_doc_path,
)
from docstra.core.services.change_detection_service import ChangeDetectionService
from docstra.core.services.ingestion_service import IngestionService
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

    def _resolve_persist_directory(self, input_path_abs: Path) -> Path:
        """Resolve the persist directory relative to the codebase path."""
        base_persist_path = Path(self.user_config.storage.persist_directory)
        if not base_persist_path.is_absolute():
            base_persist_path = input_path_abs / base_persist_path
        return base_persist_path.resolve()

    def _load_code_index(
        self, abs_persist_directory: Path, input_path_abs: Path
    ) -> Optional[CodebaseIndex]:
        """Load the core index if a manifest exists."""
        index_dir = abs_persist_directory / "index"
        if not (index_dir / CORE_INDEX_FILENAME).exists():
            return None
        try:
            return CodebaseIndexer(
                index_directory=str(index_dir),
                codebase_root=str(input_path_abs),
            ).get_index()
        except Exception as e:
            self.console.print(
                f"[yellow]Warning: Could not load code index: {e}[/yellow]"
            )
            return None

    def _load_documents(
        self,
        input_path_abs: Path,
        include_dirs: List[str],
        exclude_patterns: List[str],
    ) -> List[Document]:
        """Collect, filter, and parse the documents to be documented."""
        all_files = collect_files(
            base_path=str(input_path_abs),
            file_extensions=FileCollector.default_code_file_extensions(),
        )
        target_file_paths = filter_files_with_patterns(
            file_paths=all_files,
            base_path=str(input_path_abs),
            include_dirs=include_dirs,
            exclude_patterns=exclude_patterns,
        )

        documents: List[Document] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            task_load = progress.add_task(
                "[cyan]Loading files...", total=len(target_file_paths)
            )
            for file_path in target_file_paths:
                try:
                    document = self.document_processor.process(str(file_path))
                    if document:
                        document.metadata.filepath = str(
                            file_path.relative_to(input_path_abs)
                        )
                        documents.append(document)
                except Exception as e:
                    self.console.print(
                        f"[yellow]Failed to process {file_path}: {e}[/yellow]"
                    )
                progress.update(task_load, advance=1)

        return documents

    def _create_generator(
        self,
        input_path_abs: Path,
        abs_persist_directory: Path,
        output_dir: Path,
        project_name: str,
        project_description: str,
        style_guide: Optional[str],
        max_workers: int,
    ) -> "tuple[Any, Optional[CodebaseIndex]]":
        """Build the documentation generator plus the code index it uses."""
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

        code_index = self._load_code_index(abs_persist_directory, input_path_abs)

        if code_index:
            try:
                repo_map = RepositoryMap(str(input_path_abs), code_index)
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

        from docstra.core.documentation.generator import DocumentationGenerator

        doc_generator = DocumentationGenerator(
            llm_client=self.llm_client,
            output_dir=output_dir,
            repo_map=repo_map,
            chroma_retriever=chroma_retriever,
            code_index=code_index,
            project_name=project_name,
            project_description=project_description,
            console=self.console,
            max_workers=max_workers,
            documentation_depth="comprehensive",
            style_guide=style_guide,
            persist_directory=abs_persist_directory,
            user_config=self.user_config,
        )
        return doc_generator, code_index

    def _record_page_dependencies(
        self, documents: List[Document], code_index: Optional[CodebaseIndex]
    ) -> None:
        """Record which source files invalidate each generated file page."""
        if not self.dependency_tracker:
            return
        pages: Dict[str, List[str]] = {}
        for document in documents:
            file_id = doc_relative_path(document.metadata.filepath, code_index)
            sources = [file_id]
            if code_index:
                refs = code_index.get_file_cross_references(file_id)
                sources.extend(refs.get("imports", []))
                sources.extend(refs.get("imported_by", []))
            pages[file_doc_path(document.metadata.filepath, code_index)] = sources
        self.dependency_tracker.record_pages(pages)

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

        effective_output_dir.mkdir(parents=True, exist_ok=True)

        abs_persist_directory = self._resolve_persist_directory(input_path_abs)

        # Initialize incremental documentation components
        self.dependency_tracker = DocumentationDependencyTracker(
            str(abs_persist_directory)
        )
        self.override_manager = DocumentationOverrideManager(str(abs_persist_directory))
        self.change_detector = ChangeDetectionService(str(abs_persist_directory))

        self.console.print(
            f"[dim]Collecting files from {input_path_abs}, persist_dir for ignores: {abs_persist_directory}[/dim]"
        )
        documents_for_generation = self._load_documents(
            input_path_abs, effective_include_dirs, effective_exclude_patterns
        )
        self.console.print(
            f"Loaded {len(documents_for_generation)} files for documentation."
        )

        if not documents_for_generation:
            self.console.print(
                "[bold red]No documents could be successfully loaded for generation. Exiting.[/bold red]"
            )
            return False

        doc_generator, code_index = self._create_generator(
            input_path_abs=input_path_abs,
            abs_persist_directory=abs_persist_directory,
            output_dir=effective_output_dir,
            project_name=effective_project_name,
            project_description=effective_project_description,
            style_guide=effective_llm_style_prompt,
            max_workers=effective_max_workers,
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
                self._record_page_dependencies(documents_for_generation, code_index)
                if self.change_detector:
                    self.change_detector.mark_generation_complete(str(input_path_abs))
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
    ) -> bool:
        """Regenerate only the documentation impacted by source changes.

        Detects changed files, rebuilds the core index, computes the impacted
        set from the import graph (a page depends on its file and that file's
        graph neighbors in both directions), and regenerates only those file
        and module pages.

        Args:
            input_path_str: Path to the codebase
            output_dir_str: Output directory for documentation
            base_ref: Git reference to compare against for changes
            force_files: List of files to force regeneration

        Returns:
            True if successful, False otherwise
        """
        input_path_abs = Path(input_path_str).resolve()
        abs_persist_directory = self._resolve_persist_directory(input_path_abs)

        self.dependency_tracker = DocumentationDependencyTracker(
            str(abs_persist_directory)
        )
        self.override_manager = DocumentationOverrideManager(str(abs_persist_directory))
        self.change_detector = ChangeDetectionService(str(abs_persist_directory))

        self.console.print(
            Panel("Incremental Documentation Update", style="bold green")
        )

        # Detect changes
        if force_files:
            change_analysis = self.change_detector.detect_changes_from_file_list(
                force_files
            )
            self.console.print(
                f"[yellow]Forced regeneration for {len(force_files)} files[/yellow]"
            )
        else:
            change_analysis = self.change_detector.detect_changes_from_git(
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

        # Filter changed files through overrides before computing impact
        changed_file_paths = [
            file_path
            for file_path in change_analysis.changed_files + change_analysis.new_files
            if not self.override_manager.should_skip_generation(file_path)
        ]

        # Snapshot the pre-change graph, then rebuild the index so
        # cross-references reflect the current code.
        old_index = self._load_code_index(abs_persist_directory, input_path_abs)
        ingestion = IngestionService(console=self.console)
        if not ingestion.build_core_index(str(input_path_abs), self.user_config):
            self.console.print("[bold red]Core index rebuild failed.[/bold red]")
            return False
        new_index = self._load_code_index(abs_persist_directory, input_path_abs)

        def to_file_id(path: str) -> str:
            if new_index:
                return new_index.normalize_file_id(path)
            try:
                return Path(path).resolve().relative_to(input_path_abs).as_posix()
            except ValueError:
                return path

        graphs = [index for index in (old_index, new_index) if index]
        changed_ids = {to_file_id(path) for path in changed_file_paths}
        deleted_ids = {to_file_id(path) for path in change_analysis.deleted_files}

        impacted = compute_impacted_file_ids(changed_ids, graphs)
        if deleted_ids and old_index:
            impacted |= compute_impacted_file_ids(deleted_ids, [old_index])
        impacted -= deleted_ids

        effective_output_dir = Path(output_dir_str or self.doc_config.output_dir)

        # Remove pages for deleted source files
        for deleted_id in sorted(deleted_ids):
            stale_page = effective_output_dir / "docs" / "api" / f"{deleted_id}.md"
            if stale_page.exists():
                stale_page.unlink()
                self.console.print(f"[dim]Removed stale page for {deleted_id}[/dim]")

        self.console.print(
            f"[cyan]Impacted files (changed plus graph neighbors): {len(impacted)}[/cyan]"
        )

        include_dirs = self.doc_config.include_dirs or []
        exclude_patterns = self.doc_config.exclude_patterns or []
        documents = self._load_documents(input_path_abs, include_dirs, exclude_patterns)
        if not documents:
            self.console.print("[yellow]No documents to process.[/yellow]")
            return True

        effective_output_dir.mkdir(parents=True, exist_ok=True)
        doc_generator, code_index = self._create_generator(
            input_path_abs=input_path_abs,
            abs_persist_directory=abs_persist_directory,
            output_dir=effective_output_dir,
            project_name=self.doc_config.project_name or input_path_abs.name,
            project_description=self.doc_config.project_description or "",
            style_guide=self.doc_config.llm_style_prompt,
            max_workers=max(1, self.doc_config.max_workers_default or 2),
        )

        success = doc_generator.update_documentation(documents, impacted)

        if success:
            impacted_documents = [
                doc
                for doc in documents
                if doc_relative_path(doc.metadata.filepath, code_index) in impacted
            ]
            self._record_page_dependencies(impacted_documents, code_index)
            self.change_detector.mark_generation_complete(str(input_path_abs))
            self.console.print(
                "[bold green]Incremental documentation update completed successfully![/bold green]"
            )
        else:
            self.console.print(
                "[bold red]Incremental documentation update failed.[/bold red]"
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
