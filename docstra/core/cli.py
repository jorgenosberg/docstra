# File: ./docstra/core/cli.py

"""
Command-line interface for the code documentation assistant.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import List, Optional, Union, Dict, Any, Callable, cast

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Confirm
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.table import Table
from rich.text import Text
from rich.align import Align

from docstra.core.config.settings import (
    ConfigManager,
    ModelProvider,
    ProcessingConfig,
    UserConfig,
)
from docstra.core.document_processing.extractor import DocumentProcessor
from docstra.core.documentation.generator import DocumentationGenerator
from docstra.core.indexing.code_index import CodebaseIndex, CodebaseIndexer
from docstra.core.indexing.model import CORE_INDEX_FILENAME
from docstra.core.ingestion.embeddings import EmbeddingFactory
from docstra.core.ingestion.storage import ChromaDBStorage
from docstra.core.llm.anthropic import AnthropicClient
from docstra.core.llm.local import LocalModelClient
from docstra.core.llm.ollama import OllamaClient
from docstra.core.llm.openai import OpenAIClient
from docstra.core.retrieval.chroma import ChromaRetriever
from docstra.core.retrieval.evaluation import (
    DEFAULT_RETRIEVAL_EVAL_CASES,
    RetrievalEvalSummary,
    evaluate_retrieval_cases,
)
from docstra.core.ingestion.fts_storage import FtsStorage
from docstra.core.retrieval.fts import FtsRetriever
from docstra.core.retrieval.fusion import FusionRetriever
from docstra.core.services.initialization_service import InitializationService
from docstra.core.services.ingestion_service import IngestionService
from docstra.core.services.query_service import QueryService
from docstra.core.services.chat_service import ChatService
from docstra.core.services.documentation_service import DocumentationService
from docstra.core.services.config_service import ConfigService
from docstra.core.tracking.llm_tracker import UniversalLLMTracker
from docstra.core.utils.language_detector import LanguageDetector
from urllib.parse import quote
import re
from pathlib import Path
from docstra.core.utils.colors import Colors

RETRIEVAL_EVAL_CANDIDATE_MULTIPLIER = 5
RETRIEVAL_EVAL_MIN_CANDIDATES = 50


def display_docstra_header() -> None:
    """Display the DOCSTRA ASCII art header with subtle styling."""
    ascii_art = """
██████╗  ██████╗  ██████╗███████╗████████╗██████╗  █████╗ 
██╔══██╗██╔═══██╗██╔════╝██╔════╝╚══██╔══╝██╔══██╗██╔══██╗
██║  ██║██║   ██║██║     ███████╗   ██║   ██████╔╝███████║
██║  ██║██║   ██║██║     ╚════██║   ██║   ██╔══██╗██╔══██║
██████╔╝╚██████╔╝╚██████╗███████║   ██║   ██║  ██║██║  ██║
╚═════╝  ╚═════╝  ╚═════╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝
"""

    # Use Rich's default styling - much cleaner
    header_text = Text(ascii_art.strip(), style=Colors.HIGHLIGHT_BOLD)
    tagline = Text("LLM-Powered Code Documentation Assistant", style=Colors.DIM)

    # Center the header and tagline
    console.print()
    console.print(Align.center(header_text))
    console.print(Align.center(tagline))
    console.print()


def serve_documentation(docs_dir: str, port: int = 8000) -> None:
    """Serve documentation using a simple HTTP server."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn

    app = FastAPI(title="Documentation Server")

    # Mount static files
    app.mount(
        "/static",
        StaticFiles(directory=os.path.join(docs_dir, "static")),
        name="static",
    )

    @app.get("/", response_class=HTMLResponse)
    async def read_index() -> HTMLResponse:
        index_path = os.path.join(docs_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                return HTMLResponse(f.read())
        else:
            # Try other formats
            md_path = os.path.join(docs_dir, "index.md")
            if os.path.exists(md_path):
                # Convert markdown to HTML
                import markdown

                with open(md_path, "r") as f:
                    content = f.read()
                html_content = markdown.markdown(content)
                return HTMLResponse(
                    f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Documentation</title>
                    <link rel="stylesheet" href="/static/style.css">
                </head>
                <body>
                    <div class="container">
                        {html_content}
                    </div>
                </body>
                </html>
                """
                )

            # If all else fails
            raise HTTPException(status_code=404, detail="Index not found")

    @app.get("/{path:path}")
    async def read_file(path: str) -> Union[FileResponse, HTMLResponse]:
        full_path = os.path.join(docs_dir, path)
        if os.path.exists(full_path):
            return FileResponse(full_path)

        # Try with extensions
        for ext in [".html", ".md", ".rst"]:
            if os.path.exists(full_path + ext):
                if ext == ".md":
                    # Convert markdown to HTML
                    import markdown

                    with open(full_path + ext, "r") as f:
                        content = f.read()
                    html_content = markdown.markdown(content)
                    return HTMLResponse(
                        f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Documentation</title>
                        <link rel="stylesheet" href="/static/style.css">
                    </head>
                    <body>
                        <div class="container">
                            {html_content}
                        </div>
                    </body>
                    </html>
                    """
                    )
                else:
                    return FileResponse(full_path + ext)

        raise HTTPException(status_code=404, detail=f"File {path} not found")

    console.print(
        f"[{Colors.SUCCESS_BOLD}]Starting documentation server at:[/] http://localhost:{port}"
    )
    uvicorn.run(app, host="0.0.0.0", port=port)


# Initialize typer app
app = typer.Typer(
    name="docstra",
    help="LLM-powered code documentation assistant with repository exploration and analysis",
    add_completion=False,
)

# Initialize rich console
console = Console()


def get_llm_client(
    config_manager: ConfigManager,
) -> Union[AnthropicClient, OpenAIClient, OllamaClient, LocalModelClient]:
    """Get the appropriate LLM client based on configuration.

    Args:
        config_manager: Configuration manager

    Returns:
        LLM client
    """
    config = config_manager.config
    provider = config.model.provider

    if provider == ModelProvider.ANTHROPIC:
        return AnthropicClient(
            model_name=config.model.model_name,
            api_key=config.model.api_key,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
        )
    elif provider == ModelProvider.OPENAI:
        return OpenAIClient(
            model_name=config.model.model_name,
            api_key=config.model.api_key,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
        )
    elif provider == ModelProvider.OLLAMA:
        return OllamaClient(
            model_name=config.model.model_name,
            api_base=config.model.api_base or "http://localhost:11434",
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
            validate_connection=False,  # Don't validate during CLI operations
        )
    elif provider == ModelProvider.LOCAL:
        return LocalModelClient(
            model_name=config.model.model_name,
            model_path=config.model.model_path,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
            device=config.model.device,
        )
    else:
        raise ValueError(f"Unsupported model provider: {provider}")


@app.command()
def init(
    codebase_path: str = typer.Argument(".", help="Path to the codebase to document"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to the configuration file"
    ),
    exclude: List[str] = typer.Option(
        [], "--exclude", "-e", help="Patterns to exclude from processing"
    ),
    include: List[str] = typer.Option(
        [], "--include", "-i", help="Directories to specifically include"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force reindexing of the codebase"
    ),
    wizard: bool = typer.Option(
        True, "--wizard/--no-wizard", help="Run interactive configuration wizard"
    ),
) -> None:
    """Initialize the code documentation assistant for a codebase."""

    # Display beautiful header for init command
    display_docstra_header()

    # Detect if any options (other than the default positional argument) were provided
    provided_options = [
        opt
        for opt in [
            (config_path, "--config"),
            (exclude, "--exclude"),
            (include, "--include"),
            (force, "--force"),
            (wizard is False, "--no-wizard"),
        ]
        if (opt[0] and opt[1] != "--no-wizard") or (opt[1] == "--no-wizard" and opt[0])
    ]
    # If no options were provided, run the wizard (UX-friendly default)
    run_wizard = False
    if not provided_options:
        run_wizard = True
    # If --no-wizard is explicitly set, never run the wizard
    if wizard is False:
        run_wizard = False

    # Initialize configuration
    config_manager = ConfigManager(config_path)

    # Always pass exclude patterns to initialization service so they are written to .docstraignore
    from docstra.core.services.initialization_service import InitializationService

    init_service = InitializationService(console=console)
    abs_codebase_path = os.path.abspath(codebase_path)
    init_service.initialize_project(
        codebase_path=abs_codebase_path,
        config_file_path=config_path,
        run_wizard=run_wizard,
        initial_include_patterns=include if include else None,
        initial_exclude_patterns=exclude if exclude else None,
    )

    # Reload config after initialization
    config_manager = ConfigManager(config_path)

    # Create a clean summary panel using semantic colors
    console.print("\n" + "─" * 60)
    console.print(f"[{Colors.SUCCESS_BOLD}]✓ Project initialized successfully![/]")
    console.print("─" * 60)
    console.print(f"📁 [{Colors.BOLD}]Codebase:[/] {abs_codebase_path}")
    console.print(f"⚙️  [{Colors.BOLD}]Configuration:[/] {config_manager.config_path}")
    console.print(
        f"💾 [{Colors.BOLD}]Storage:[/] {config_manager.config.storage.persist_directory}"
    )
    console.print(
        f"🤖 [{Colors.BOLD}]Model:[/] {config_manager.config.model.provider} - {config_manager.config.model.model_name}"
    )
    console.print("─" * 60)

    # Next steps with semantic colors
    console.print(f"\n[{Colors.BOLD}]📋 Next Steps:[/]")
    console.print(
        f"   1️⃣  [{Colors.HIGHLIGHT}]docstra ingest[/] - Process and index your codebase"
    )
    console.print(
        f'   2️⃣  [{Colors.HIGHLIGHT}]docstra query[/] "your question" - Ask questions about your code'
    )
    console.print(
        f"   3️⃣  [{Colors.HIGHLIGHT}]docstra chat[/] - Start an interactive chat session"
    )

    # Optionally prompt to run ingestion now - use Rich defaults
    console.print()
    if Confirm.ask(
        "🚀 Would you like to ingest and index your codebase now?", default=False
    ):
        console.print()  # Add spacing before ingestion
        ingest(
            codebase_path=abs_codebase_path,
            config_path=config_path,
            exclude=exclude,
            include=include,
            force=force,
        )
    else:
        console.print(
            f"[{Colors.DIM}]💡 Run [{Colors.HIGHLIGHT}]docstra ingest[/] when ready to process your codebase[/]"
        )


@app.command()
def document(
    file_path: str = typer.Argument(..., help="Path to the file to document"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to the configuration file"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Path to save the generated documentation"
    ),
) -> None:
    """Generate documentation for a file."""
    # Initialize configuration
    config_manager = ConfigManager(config_path)

    # Check if file exists
    if not os.path.exists(file_path):
        console.print(f"[{Colors.ERROR_BOLD}]Error:[/] File {file_path} does not exist")
        sys.exit(1)

    # Process the file
    console.print(f"Generating documentation for [{Colors.BOLD}]{file_path}[/]")

    # Initialize document processor
    doc_processor = DocumentProcessor()

    # Process the file
    document = doc_processor.process(file_path)

    # Get language
    language = str(document.metadata.language)

    # Get LLM client
    llm_client = get_llm_client(config_manager)

    # Generate documentation
    with console.status(f"[{Colors.INFO}]Generating documentation...", spinner="dots"):
        documentation = llm_client.document_code(
            code=document.content,
            language=language,
            additional_context=f"File path: {file_path}",
        )
        # Ensure documentation is a string
        documentation_str = str(documentation)

    # Output the documentation
    if output_file:
        with open(output_file, "w") as f:
            f.write(documentation_str)
        console.print(f"Documentation saved to [{Colors.BOLD}]{output_file}[/]")
    else:
        console.print(
            Panel(
                Markdown(documentation_str),
                title=f"Documentation for {os.path.basename(file_path)}",
            )
        )


@app.command()
def explain(
    file_path: str = typer.Argument(..., help="Path to the file to explain"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to the configuration file"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Path to save the explanation"
    ),
) -> None:
    """Explain a file."""
    # Initialize configuration
    config_manager = ConfigManager(config_path)

    # Check if file exists
    if not os.path.exists(file_path):
        console.print(f"[{Colors.ERROR_BOLD}]Error:[/] File {file_path} does not exist")
        sys.exit(1)

    # Process the file
    console.print(f"Generating explanation for [{Colors.BOLD}]{file_path}[/]")

    # Initialize document processor
    doc_processor = DocumentProcessor()

    # Process the file
    document = doc_processor.process(file_path)

    # Get language
    language = str(document.metadata.language)

    # Get LLM client
    llm_client = get_llm_client(config_manager)

    # Generate explanation
    with console.status(f"[{Colors.INFO}]Generating explanation...", spinner="dots"):
        explanation = llm_client.explain_code(
            code=document.content,
            language=language,
            additional_context=f"File path: {file_path}",
        )
        # Ensure explanation is a string
        explanation_str = str(explanation)

    # Output the explanation
    if output_file:
        with open(output_file, "w") as f:
            f.write(explanation_str)
        console.print(f"Explanation saved to [{Colors.BOLD}]{output_file}[/]")
    else:
        console.print(
            Panel(
                Markdown(explanation_str),
                title=f"Explanation for {os.path.basename(file_path)}",
            )
        )


@app.command()
def examples(
    query: str = typer.Argument(..., help="What kind of code examples to generate"),
    language: str = typer.Argument(..., help="Programming language for the examples"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to the configuration file"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Path to save the generated examples"
    ),
) -> None:
    """Generate code examples."""
    # Initialize configuration
    config_manager = ConfigManager(config_path)

    # Get LLM client
    llm_client = get_llm_client(config_manager)

    # Generate examples
    console.print(f"Generating {language} code examples for: [{Colors.BOLD}]{query}[/]")

    with console.status(f"[{Colors.INFO}]Generating examples...", spinner="dots"):
        examples = llm_client.generate_examples(request=query, language=language)
        # Ensure examples is a string
        examples_str = str(examples)

    # Output the examples
    if output_file:
        with open(output_file, "w") as f:
            f.write(examples_str)
        console.print(f"Examples saved to [{Colors.BOLD}]{output_file}[/]")
    else:
        console.print(
            Panel(Markdown(examples_str), title=f"{language} Examples for {query}")
        )


@app.command()
def config(
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to the configuration file"
    ),
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    reset: bool = typer.Option(False, "--reset", help="Reset configuration to default"),
    set_model: Optional[str] = typer.Option(
        None, "--model", help="Set model provider (anthropic, openai, ollama, local)"
    ),
    set_model_name: Optional[str] = typer.Option(
        None, "--model-name", help="Set model name"
    ),
    set_embedding: Optional[str] = typer.Option(
        None, "--embedding", help="Set embedding provider (huggingface, openai, ollama)"
    ),
) -> None:
    """Manage configuration."""
    # Initialize configuration manager
    config_manager = ConfigManager(config_path)

    # Reset configuration if requested
    if reset:
        config_manager.reset_to_default()
        console.print(f"[{Colors.SUCCESS_BOLD}]Configuration reset to default[/]")

    # Update configuration if needed
    changes = False

    if set_model:
        try:
            provider = ModelProvider(set_model.lower())
            config_manager.update(model={"provider": provider})
            changes = True
        except ValueError:
            console.print(
                f"[{Colors.ERROR_BOLD}]Error:[/] Invalid model provider: {set_model}"
            )
            console.print("Available providers: anthropic, openai, ollama, local")

    if set_model_name:
        config_manager.update(model={"model_name": set_model_name})
        changes = True

    if set_embedding:
        config_manager.update(embedding={"provider": set_embedding})
        changes = True

    if changes:
        console.print(f"[{Colors.SUCCESS_BOLD}]Configuration updated[/]")

    # Show configuration - use semantic colors consistently
    if show or (not reset and not changes):
        config = config_manager.config

        console.print(f"[{Colors.BOLD}]Current Configuration:[/]")
        console.print(
            f"Config path: [{Colors.HIGHLIGHT}]{config_manager.config_path}[/]"
        )
        console.print(f"\n[{Colors.BOLD}]Model:[/]")
        console.print(f"  Provider: [{Colors.HIGHLIGHT}]{config.model.provider}[/]")
        console.print(f"  Model name: [{Colors.HIGHLIGHT}]{config.model.model_name}[/]")
        console.print(
            f"  Temperature: [{Colors.HIGHLIGHT}]{config.model.temperature}[/]"
        )
        console.print(f"  Max tokens: [{Colors.HIGHLIGHT}]{config.model.max_tokens}[/]")

        console.print(f"\n[{Colors.BOLD}]Embedding:[/]")
        console.print(f"  Provider: [{Colors.HIGHLIGHT}]{config.embedding.provider}[/]")
        console.print(
            f"  Model name: [{Colors.HIGHLIGHT}]{config.embedding.model_name}[/]"
        )

        console.print(f"\n[{Colors.BOLD}]Storage:[/]")
        console.print(
            f"  Persist directory: [{Colors.HIGHLIGHT}]{config.storage.persist_directory}[/]"
        )

        console.print(f"\n[{Colors.BOLD}]Processing:[/]")
        console.print(
            f"  Chunk size: [{Colors.HIGHLIGHT}]{config.processing.chunk_size}[/]"
        )
        console.print(
            f"  Chunk overlap: [{Colors.HIGHLIGHT}]{config.processing.chunk_overlap}[/]"
        )
        console.print("  Exclude patterns:")
        for pattern in config.processing.exclude_patterns:
            console.print(f"    - [{Colors.HIGHLIGHT}]{pattern}[/]")


def parse_start_end(line_spec: str) -> tuple[int, int]:
    """Parse a line specification (e.g., '10-20')."""
    if "-" in line_spec:
        start, end = line_spec.split("-", 1)
        return int(start), int(end)
    else:
        line = int(line_spec)
        return line, line


@app.command()
def analyze(
    file_path: str = typer.Argument(..., help="Path to the file to analyze"),
    lines: Optional[str] = typer.Option(
        None, "--lines", "-l", help="Line range to analyze (e.g. '10-20')"
    ),
    context: bool = typer.Option(
        False, "--context", help="Include repository context in analysis"
    ),
    impact: bool = typer.Option(False, "--impact", help="Show change impact analysis"),
    context_mode: Optional[str] = typer.Option(
        None, "--context-mode", help="Context mode: compact, balanced, detailed"
    ),
    complexity: bool = typer.Option(
        False, "--complexity", help="Show detailed complexity metrics"
    ),
    relationships: bool = typer.Option(
        False, "--relationships", help="Show file relationships"
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to the configuration file"
    ),
) -> None:
    """Analyze a specific part of a file with optional repository context."""
    # Initialize configuration
    config_manager = ConfigManager(config_path)

    # Check if file exists
    if not os.path.exists(file_path):
        console.print(f"[{Colors.ERROR_BOLD}]Error:[/] File {file_path} does not exist")
        sys.exit(1)

    # Process the file
    console.print(f"Analyzing [{Colors.BOLD}]{file_path}[/]")

    # Initialize document processor
    doc_processor = DocumentProcessor()

    # Process the file
    document = doc_processor.process(file_path)

    # Get language
    language = str(document.metadata.language)

    # Extract specified lines if provided
    start_line, end_line = 1, document.metadata.line_count
    if lines:
        try:
            start_line, end_line = parse_start_end(lines)

            if start_line < 1 or end_line > document.metadata.line_count:
                console.print(
                    f"[{Colors.ERROR_BOLD}]Error:[/] Line range {start_line}-{end_line} is out of bounds (1-{document.metadata.line_count})"
                )
                sys.exit(1)

            content_lines = document.content.splitlines()
            code_to_analyze = "\n".join(content_lines[start_line - 1 : end_line])
        except Exception:
            console.print(f"[{Colors.ERROR_BOLD}]Error:[/] Invalid line range: {lines}")
            sys.exit(1)
    else:
        code_to_analyze = document.content

    # Get repository context if requested
    repo_context = ""
    if context or impact or complexity or relationships:
        try:
            from docstra.core.services.repository_explorer_service import (
                RepositoryExplorerService,
            )

            user_config = load_or_init_config(config_path)

            # Apply context mode override if provided
            if context_mode:
                if context_mode not in ["compact", "balanced", "detailed"]:
                    console.print(
                        f"[{Colors.ERROR_BOLD}]Error: Invalid context mode '{context_mode}'. Must be: compact, balanced, detailed[/]"
                    )
                    sys.exit(1)
                user_config.model.context_mode = context_mode

            explorer_service = RepositoryExplorerService(user_config, console)

            file_relationships = explorer_service.get_file_relationships(file_path)

            # Build context information
            context_parts = []

            if context:
                context_parts.append(
                    f"Repository Context for {os.path.basename(file_path)}:"
                )
                context_parts.append(
                    f"- Module type: {file_relationships['complexity_info'].get('module_type', 'Unknown')}"
                )
                context_parts.append(
                    f"- Dependencies: {len(file_relationships['dependencies'])} files"
                )
                context_parts.append(
                    f"- Dependents: {len(file_relationships['dependents'])} files"
                )
                context_parts.append(
                    f"- Related files: {len(file_relationships['related_files'])} files"
                )

            if complexity:
                complexity_info = file_relationships.get("complexity_info", {})
                arch_info = file_relationships.get("architectural_info", {})
                context_parts.append("Complexity Information:")
                context_parts.append(
                    f"- File complexity: {complexity_info.get('complexity', 'N/A')}"
                )
                context_parts.append(
                    f"- File size: {complexity_info.get('size_kb', 0):.1f} KB"
                )
                context_parts.append(
                    f"- Centrality score: {arch_info.get('centrality_score', 0)}"
                )
                context_parts.append(
                    f"- Is core module: {arch_info.get('is_core_module', False)}"
                )

            if relationships:
                context_parts.append("Relationships:")
                if file_relationships["dependencies"]:
                    context_parts.append(
                        f"- Imports from: {', '.join([os.path.basename(d) for d in file_relationships['dependencies'][:5]])}"
                    )
                if file_relationships["dependents"]:
                    context_parts.append(
                        f"- Used by: {', '.join([os.path.basename(d) for d in file_relationships['dependents'][:5]])}"
                    )

            repo_context = "\n".join(context_parts)

        except Exception as e:
            console.print(
                f"[{Colors.WARNING}]Warning: Could not load repository context: {e}[/]"
            )
            repo_context = ""

    # Get LLM client
    llm_client = get_llm_client(config_manager)

    # Build additional context
    additional_context = f"File path: {file_path}, Lines: {start_line}-{end_line}"
    if repo_context:
        additional_context += f"\n\n{repo_context}"

    # Analyze the code
    with console.status(f"[{Colors.INFO}]Analyzing code...", spinner="dots"):
        analysis = llm_client.explain_code(
            code=code_to_analyze,
            language=language,
            additional_context=additional_context,
        )
        # Ensure analysis is a string
        analysis_str = str(analysis)

    # Output the analysis
    line_info = f" (lines {start_line}-{end_line})" if lines else ""
    console.print(
        Panel(
            Markdown(analysis_str),
            title=f"Analysis of {os.path.basename(file_path)}{line_info}",
        )
    )

    # Show additional repository information if requested
    if (
        context or impact or complexity or relationships
    ) and "explorer_service" in locals():
        try:
            if impact and explorer_service.repo_map:
                # Show change impact analysis
                console.print(f"\n[{Colors.BOLD}]Change Impact Analysis:[/]")
                impact_map = explorer_service.repo_map.get_change_impact_analysis(
                    [file_path]
                )
                impacted_files = impact_map.get(file_path, [])

                if impacted_files:
                    impact_table = Table(
                        title="Files that would be impacted by changes",
                        show_header=True,
                        header_style="bold orange",
                    )
                    impact_table.add_column("Impacted File", style="orange")
                    impact_table.add_column("Impact Type", style="yellow")

                    for impacted_file in impacted_files[:10]:
                        impact_table.add_row(
                            os.path.basename(impacted_file),
                            "Direct/Indirect dependency",
                        )

                    console.print(impact_table)

                    if len(impacted_files) > 10:
                        console.print(
                            f"[{Colors.DIM}]... and {len(impacted_files) - 10} more files[/]"
                        )
                else:
                    console.print(
                        f"[{Colors.SUCCESS}]No files would be directly impacted by changes to this file[/]"
                    )

            if relationships and file_relationships:
                # Display detailed relationships
                console.print(f"\n[{Colors.BOLD}]File Relationships:[/]")
                explorer_service.display_file_relationships(file_relationships)

        except Exception as e:
            console.print(
                f"[{Colors.WARNING}]Warning: Could not display additional analysis: {e}[/]"
            )


@app.command()
def generate(
    path: str = typer.Argument(
        ".", help="File or directory to generate documentation for"
    ),
    output_dir: str = typer.Option(
        None, "--output", "-o", help="Output directory for documentation"
    ),
    format: str = typer.Option(
        None, "--format", "-f", help="Output format (html, markdown, rst)"
    ),
    serve: bool = typer.Option(
        False, "--serve", "-s", help="Serve documentation after generation"
    ),
    port: int = typer.Option(
        8000, "--port", "-p", help="Port to serve documentation on"
    ),
    context_mode: Optional[str] = typer.Option(
        None, "--context-mode", help="Context mode: compact, balanced, detailed"
    ),
    wizard: bool = typer.Option(
        True, "--wizard/--no-wizard", help="Run interactive configuration wizard"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", help="Project name for documentation"
    ),
    description: Optional[str] = typer.Option(
        None, "--description", help="Project description"
    ),
    exclude: List[str] = typer.Option(
        [], "--exclude", "-e", help="Patterns to exclude from documentation"
    ),
    include: List[str] = typer.Option(
        [], "--include", "-i", help="Directories to specifically include"
    ),
    theme: Optional[str] = typer.Option(
        None, "--theme", "-t", help="Documentation theme"
    ),
    use_saved_config: bool = typer.Option(
        False, "--use-saved", help="Use previously saved configuration"
    ),
) -> None:
    """Generate comprehensive documentation for a file or directory."""
    from docstra.core.documentation.wizard import run_documentation_wizard
    from docstra.core.utils.file_collector import collect_files, FileCollector

    # Initialize config with default values
    config = {
        "name": os.path.basename(os.path.abspath(path)),
        "description": "",
        "version": "0.1.0",
        "include_dirs": [],
        # Create ProcessingConfig with required parameters
        "exclude_dirs": ProcessingConfig(
            chunk_size=800, chunk_overlap=100
        ).exclude_patterns,
        "exclude_files": [],
        "theme": "default",
        "output_dir": "./docs",
        "format": "html",
    }

    # Try to load saved config if requested
    if use_saved_config:
        # Create a helper function to simulate load_wizard_config
        # In the real implementation, this would be part of the wizard module
        def load_saved_config(path: str) -> Optional[Dict[str, Any]]:
            """Load saved configuration from a file."""
            config_path = os.path.join(path, ".docstra", "docs_config.json")
            if os.path.exists(config_path):
                import json

                with open(config_path) as f:
                    # Cast the result to Dict[str, Any] to satisfy mypy
                    return cast(Dict[str, Any], json.load(f))
            return None  # Explicitly return None

        saved_config = load_saved_config(path)
        if saved_config:
            config.update(saved_config)
            console.print(f"[{Colors.SUCCESS_BOLD}]Loaded saved configuration[/]")
        else:
            console.print(
                f"[{Colors.WARNING}]No saved configuration found, using defaults[/]"
            )

    # Override with command line arguments
    if output_dir:
        config["output_dir"] = output_dir
    if format:
        config["format"] = format
    if name:
        config["name"] = name
    if description:
        config["description"] = description
    if exclude:
        # Convert glob patterns to gitignore patterns
        config["exclude_dirs"] = [
            pattern.replace("**/", "").replace("/**", "") for pattern in exclude
        ]
    if include:
        config["include_dirs"] = include
    if theme:
        config["theme"] = theme

    # Run the wizard if requested and not overridden by CLI args
    if wizard and not (
        output_dir and format and name and description and (exclude or include)
    ):
        try:
            # Create a config manager for the wizard
            config_manager = ConfigManager()
            # Assume wizard updates config in-place and doesn't return anything
            # This is a safe assumption based on the error message
            run_documentation_wizard(console, path, config_manager)
            # Since run_documentation_wizard doesn't return a value, no update needed
        except KeyboardInterrupt:
            console.print(
                f"\n[{Colors.WARNING}]Wizard cancelled, using default/CLI values[/]"
            )
        except Exception as e:
            console.print(f"\n[{Colors.ERROR}]Error in wizard: {e}[/]")
            console.print(f"[{Colors.WARNING}]Proceeding with default/CLI values[/]")

    # Print configuration summary with enhanced styling
    console.print(
        Panel(
            f"[{Colors.BOLD}]📚 Generating Documentation for:[/] {config['name']}",
            style=Colors.INFO_BOLD,
            expand=False,
        )
    )
    console.print(f"📄 [{Colors.BOLD}]Description:[/] {config['description']}")
    console.print(f"📁 [{Colors.BOLD}]Output directory:[/] {config['output_dir']}")
    console.print(f"📄 [{Colors.BOLD}]Format:[/] {config['format']}")

    if config["include_dirs"]:
        console.print(
            f"🎯 [{Colors.BOLD}]Including directories:[/] {', '.join(config['include_dirs'])}"
        )

    console.print(
        f"🚫 [{Colors.BOLD}]Excluding directories:[/] {', '.join(config['exclude_dirs'])}"
    )

    # Create output directory - ensure string type
    output_dir_str = str(config["output_dir"])
    os.makedirs(output_dir_str, exist_ok=True)

    # Initialize components
    config_manager = ConfigManager()
    doc_processor = DocumentProcessor()

    # Use our file collection utility to gather files with clean status indicator
    # Convert to proper List[str] types
    include_dirs_list: List[str] = list(config["include_dirs"])
    exclude_dirs_list: List[str] = list(config["exclude_dirs"])
    exclude_files_list: List[str] = list(config["exclude_files"])

    with console.status(
        f"[{Colors.INFO}]🔍 Collecting files for documentation...", spinner="dots"
    ):
        file_paths = collect_files(
            base_path=path,
            include_dirs=include_dirs_list,
            exclude_dirs=exclude_dirs_list,
            exclude_files=exclude_files_list,
            file_extensions=FileCollector.default_code_file_extensions(),
        )

    # Show clean file collection summary
    console.print(f"[{Colors.DIM}]📂 File Collection Summary:[/]")
    console.print(
        f"   • [{Colors.SUCCESS}]Found {len(file_paths)} files to document[/]"
    )

    # Group files by directory for a helpful overview
    from collections import defaultdict
    from typing import DefaultDict

    files_by_dir: DefaultDict[str, int] = defaultdict(int)
    for file_path in file_paths:
        dir_name = (
            str(file_path.parent)
            if hasattr(file_path, "parent")
            else str(Path(file_path).parent)
        )
        # Simplify path display
        if dir_name == str(Path(path).resolve()):
            dir_name = "."
        else:
            try:
                dir_name = str(Path(dir_name).relative_to(Path(path).resolve()))
            except ValueError:
                pass  # Keep absolute path if relative conversion fails
        files_by_dir[dir_name] += 1

    # Show top directories with files
    if files_by_dir:
        sorted_dirs = sorted(files_by_dir.items(), key=lambda x: x[1], reverse=True)
        top_dirs = sorted_dirs[:5]  # Show top 5 directories
        for dir_name, count in top_dirs:
            console.print(f"   • [{Colors.SUCCESS}]{dir_name}:[/] {count} files")
        if len(sorted_dirs) > 5:
            remaining = sum(count for _, count in sorted_dirs[5:])
            console.print(
                f"   • [{Colors.DIM}]... and {remaining} files in other directories[/]"
            )

    # Process collected files with cleaner progress indication
    documents = []
    failed_files = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,  # Make progress bar disappear when done
    ) as progress:
        processing_task = progress.add_task(
            f"[{Colors.INFO}]📄 Processing files...", total=len(file_paths)
        )

        for file_path in file_paths:
            try:
                # Convert Path to str for document processor
                document = doc_processor.process(str(file_path))
                documents.append(document)
            except Exception as e:
                failed_files.append((file_path, str(e)))
            progress.update(processing_task, advance=1)

    # Show processing summary
    console.print(f"[{Colors.DIM}]📄 Processing Summary:[/]")
    console.print(
        f"   • [{Colors.SUCCESS}]Successfully processed {len(documents)} files[/]"
    )
    if failed_files:
        console.print(
            f"   • [{Colors.WARNING}]Failed to process {len(failed_files)} files[/]"
        )
        if len(failed_files) <= 3:  # Show details for few failures
            for file_path, error in failed_files:
                console.print(f"     - [{Colors.WARNING}]{file_path}: {error}[/]")
        else:
            console.print(f"     [{Colors.DIM}](Run with --verbose for details)[/]")

    # No files found
    if not documents:
        console.print(f"[{Colors.WARNING_BOLD}]No files found to document![/]")
        return

    # Use the documentation service for better progress reporting
    user_config = load_or_init_config()

    # Apply context mode override if provided
    if context_mode:
        if context_mode not in ["compact", "balanced", "detailed"]:
            console.print(
                f"[{Colors.ERROR_BOLD}]Error: Invalid context mode '{context_mode}'. Must be: compact, balanced, detailed[/]"
            )
            return
        user_config.model.context_mode = context_mode

    _, _, _, documentation_service = create_services_for_config(user_config)

    # Generate documentation using the service
    success = documentation_service.generate_documentation(
        input_path_str=path,
        output_dir_str=str(config["output_dir"]),
        project_name_str=str(config["name"]) if config["name"] else None,
        project_description_str=str(config["description"])
        if config["description"]
        else None,
        cli_include_patterns=include_dirs_list if include_dirs_list else None,
        cli_exclude_patterns=exclude_dirs_list if exclude_dirs_list else None,
    )

    if not success:
        console.print(f"[{Colors.ERROR_BOLD}]Documentation generation failed.[/]")
        return

    # Ensure output_dir is a string for os.path.abspath
    output_dir_abs = os.path.abspath(str(config["output_dir"]))
    console.print("\n" + "─" * 60)
    console.print(f"[{Colors.SUCCESS_BOLD}]🎉 Documentation generated successfully![/]")
    console.print("─" * 60)
    console.print(f"📁 [{Colors.BOLD}]Location:[/] {output_dir_abs}")
    console.print("─" * 60)

    # Serve documentation if requested
    if serve:
        serve_documentation_from_generator(str(config["output_dir"]), port)


# Update the serve_documentation function to use the enhanced DocumentationGenerator
def serve_documentation_from_generator(docs_dir: str, port: int = 8000) -> None:
    """Serve documentation using MkDocs or a simple HTTP server."""

    # Create a minimal generator instance just for serving
    dummy_gen = DocumentationGenerator(None, docs_dir)
    dummy_gen.serve_documentation(port)


# Helper function to load or initialize config
def load_or_init_config(config_path: Optional[str] = None) -> UserConfig:
    """Loads configuration or initializes if it doesn't exist."""
    try:
        # ConfigManager handles loading or creating the default config during init
        config_manager = ConfigManager(config_path=config_path)
        return config_manager.config
    except Exception as e:
        console.print(
            f"[{Colors.ERROR_BOLD}]Error loading or initializing configuration:[/] {e}"
        )
        raise typer.Exit(code=1)


def get_llm_tracker() -> Optional[UniversalLLMTracker]:
    """Get LLM tracker instance, creating it if needed."""
    try:
        from docstra.core.tracking.llm_tracker import get_global_tracker

        return get_global_tracker()
    except Exception as e:
        console.print(
            f"[{Colors.WARNING}]Warning: Failed to initialize LLM tracking: {e}[/]"
        )
        return None


def create_services_for_config(user_config: UserConfig) -> tuple:
    """Create service instances for the given configuration.

    Returns:
        Tuple of (ingestion_service, query_service, chat_service, documentation_service)
    """
    callbacks = None

    ingestion_service = IngestionService(console=console, callbacks=callbacks)
    query_service = QueryService(
        user_config=user_config,
        console=console,
        callbacks=callbacks,
    )
    chat_service = ChatService(
        user_config=user_config,
        console=console,
        callbacks=callbacks,
    )
    documentation_service = DocumentationService(
        user_config=user_config,
        console=console,
        callbacks=callbacks,
    )

    return ingestion_service, query_service, chat_service, documentation_service


# Initialize non-LLM services that don't require configuration
config_service = ConfigService(console=console)
init_service = InitializationService(console=console)


# New CLI commands for incremental documentation
@app.command()
def update(
    path: str = typer.Argument(
        ".", help="Path to the codebase to update documentation for"
    ),
    base_ref: str = typer.Option(
        "HEAD~1", "--base", "-b", help="Git reference to compare against for changes"
    ),
    force_files: List[str] = typer.Option(
        [], "--force", "-f", help="Force regeneration for specific files"
    ),
    output_dir: str = typer.Option(
        None, "--output", "-o", help="Output directory for documentation"
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to the configuration file"
    ),
    serve: bool = typer.Option(
        False, "--serve", "-s", help="Serve documentation after update"
    ),
    port: int = typer.Option(
        8000, "--port", "-p", help="Port to serve documentation on"
    ),
) -> None:
    """Update documentation incrementally based on detected changes."""
    display_docstra_header()

    user_config = load_or_init_config(config_path)
    console.print(
        f"[{Colors.INFO_BOLD}]Starting incremental documentation update...[/]"
    )

    try:
        doc_service = DocumentationService(user_config, console)

        success = doc_service.generate_incremental_documentation(
            input_path_str=path,
            output_dir_str=output_dir,
            base_ref=base_ref,
            force_files=force_files if force_files else None,
        )

        if success:
            console.print(
                f"[{Colors.SUCCESS_BOLD}]Documentation update completed successfully![/]"
            )

            if serve:
                output_dir_path = (
                    Path(output_dir) if output_dir else Path(path) / "docs"
                )
                serve_documentation_from_generator(str(output_dir_path), port)
        else:
            console.print(f"[{Colors.ERROR_BOLD}]Documentation update failed.[/]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[{Colors.ERROR_BOLD}]Error during documentation update: {e}[/]")
        raise typer.Exit(1)


@app.command()
def status(
    path: str = typer.Argument(".", help="Path to the codebase to check status for"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to the configuration file"
    ),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed status information"
    ),
) -> None:
    """Show documentation status and pending changes."""
    display_docstra_header()

    user_config = load_or_init_config(config_path)

    try:
        doc_service = DocumentationService(user_config, console)
        status_info = doc_service.get_documentation_status(path)

        # Display status table
        status_table = Table(
            title="Documentation Status", show_header=True, header_style="bold magenta"
        )
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="green")

        # Basic status
        status_table.add_row("Codebase Path", status_info["codebase_path"])

        if status_info["last_generation"]:
            import datetime

            last_gen_time = datetime.datetime.fromtimestamp(
                status_info["last_generation"]["timestamp"]
            )
            status_table.add_row(
                "Last Generation", last_gen_time.strftime("%Y-%m-%d %H:%M:%S")
            )
            status_table.add_row(
                "Total Files at Last Gen",
                str(status_info["last_generation"]["total_files"]),
            )
        else:
            status_table.add_row("Last Generation", "Never")

        # Current changes
        changes = status_info["current_changes"]
        status_table.add_row("Has Changes", "Yes" if changes["has_changes"] else "No")
        status_table.add_row("Total Changes", str(changes["total_changes"]))
        status_table.add_row("Changed Files", str(changes["changed_files"]))
        status_table.add_row("New Files", str(changes["new_files"]))
        status_table.add_row("Deleted Files", str(changes["deleted_files"]))

        # Dependencies and overrides
        dep_stats = status_info["dependency_stats"]
        status_table.add_row("Tracked Documents", str(dep_stats["total_docs"]))
        status_table.add_row("Total Source Files", str(dep_stats["total_source_files"]))

        override_stats = status_info["override_stats"]
        status_table.add_row("Active Overrides", str(override_stats["total"]))

        # Outdated docs
        outdated_count = len(status_info["outdated_docs"])
        status_table.add_row("Outdated Documents", str(outdated_count))

        console.print(status_table)

        if detailed:
            # Show detailed information
            if status_info["outdated_docs"]:
                console.print(f"\n[{Colors.WARNING_BOLD}]Outdated Documents:[/]")
                for doc in status_info["outdated_docs"][:10]:  # Show first 10
                    console.print(f"  • {doc}")
                if len(status_info["outdated_docs"]) > 10:
                    console.print(
                        f"  ... and {len(status_info['outdated_docs']) - 10} more"
                    )

            if override_stats["by_type"]:
                console.print(f"\n[{Colors.INFO_BOLD}]Overrides by Type:[/]")
                for override_type, count in override_stats["by_type"].items():
                    console.print(f"  • {override_type}: {count}")

            if status_info["change_history"]:
                console.print(f"\n[{Colors.INFO_BOLD}]Recent Change History:[/]")
                for i, change in enumerate(status_info["change_history"][-5:]):
                    import datetime

                    change_time = datetime.datetime.fromtimestamp(
                        change["change_timestamp"]
                    )
                    console.print(
                        f"  {i + 1}. {change_time.strftime('%Y-%m-%d %H:%M')} - {change['total_changes']} changes"
                    )

        # Suggest actions
        if changes["has_changes"]:
            console.print(
                f"\n[{Colors.HIGHLIGHT_BOLD}]💡 Suggestion:[/] Run `docstra update` to update documentation for recent changes."
            )
        else:
            console.print(f"\n[{Colors.SUCCESS}]✅ Documentation is up to date![/]")

    except Exception as e:
        console.print(
            f"[{Colors.ERROR_BOLD}]Error getting documentation status: {e}[/]"
        )
        raise typer.Exit(1)


@app.command()
def override(
    action: str = typer.Argument(..., help="Action: set, remove, list"),
    file_path: Optional[str] = typer.Argument(None, help="File path to override"),
    override_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Override type: skip, template, content"
    ),
    content: Optional[str] = typer.Option(
        None, "--content", "-c", help="Custom content or template"
    ),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="Description of the override"
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", help="Path to the configuration file"
    ),
    codebase_path: str = typer.Option(".", "--codebase", help="Path to the codebase"),
) -> None:
    """Manage documentation generation overrides for specific files."""
    display_docstra_header()

    user_config = load_or_init_config(config_path)

    # Setup persist directory
    input_path_abs = Path(codebase_path).resolve()
    persist_dir_name = user_config.storage.persist_directory
    base_persist_path = Path(persist_dir_name)
    if not base_persist_path.is_absolute():
        base_persist_path = input_path_abs / persist_dir_name
    abs_persist_directory = base_persist_path.resolve()

    from docstra.core.documentation.overrides import DocumentationOverrideManager

    override_manager = DocumentationOverrideManager(str(abs_persist_directory))

    try:
        if action == "set":
            if not file_path or not override_type:
                console.print(
                    f"[{Colors.ERROR_BOLD}]Error: file_path and --type are required for 'set' action[/]"
                )
                raise typer.Exit(1)

            # Resolve file path
            if not Path(file_path).is_absolute():
                file_path = str((input_path_abs / file_path).resolve())

            if override_type == "skip":
                override_manager.set_skip_override(file_path, description)
                console.print(
                    f"[{Colors.SUCCESS}]✅ Set skip override for {file_path}[/]"
                )
            elif override_type == "template":
                if not content:
                    console.print(
                        f"[{Colors.ERROR_BOLD}]Error: --content is required for template override[/]"
                    )
                    raise typer.Exit(1)
                override_manager.set_template_override(file_path, content, description)
                console.print(
                    f"[{Colors.SUCCESS}]✅ Set template override for {file_path}[/]"
                )
            elif override_type == "content":
                if not content:
                    console.print(
                        f"[{Colors.ERROR_BOLD}]Error: --content is required for content override[/]"
                    )
                    raise typer.Exit(1)
                override_manager.set_manual_content_override(
                    file_path, content, description
                )
                console.print(
                    f"[{Colors.SUCCESS}]✅ Set content override for {file_path}[/]"
                )
            else:
                console.print(
                    f"[{Colors.ERROR_BOLD}]Error: Invalid override type. Use: skip, template, content[/]"
                )
                raise typer.Exit(1)

        elif action == "remove":
            if not file_path:
                console.print(
                    f"[{Colors.ERROR_BOLD}]Error: file_path is required for 'remove' action[/]"
                )
                raise typer.Exit(1)

            # Resolve file path
            if not Path(file_path).is_absolute():
                file_path = str((input_path_abs / file_path).resolve())

            if override_manager.remove_override(file_path):
                console.print(
                    f"[{Colors.SUCCESS}]✅ Removed override for {file_path}[/]"
                )
            else:
                console.print(
                    f"[{Colors.WARNING}]⚠️  No override found for {file_path}[/]"
                )

        elif action == "list":
            overrides = override_manager.list_overrides(override_type)

            if not overrides:
                console.print(f"[{Colors.INFO}]No overrides found.[/]")
                return

            # Display overrides table
            override_table = Table(
                title="Documentation Overrides",
                show_header=True,
                header_style="bold magenta",
            )
            override_table.add_column("File Path", style="cyan")
            override_table.add_column("Type", style="yellow")
            override_table.add_column("Description", style="white")
            override_table.add_column("Created", style="dim")

            for override in overrides:
                import datetime

                created_time = datetime.datetime.fromtimestamp(override.created_at)
                override_table.add_row(
                    override.file_path,
                    override.override_type,
                    override.description or "No description",
                    created_time.strftime("%Y-%m-%d %H:%M"),
                )

            console.print(override_table)

        else:
            console.print(
                f"[{Colors.ERROR_BOLD}]Error: Invalid action. Use: set, remove, list[/]"
            )
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[{Colors.ERROR_BOLD}]Error managing overrides: {e}[/]")
        raise typer.Exit(1)


# Add ingest command - separate from init according to refactoring plan
@app.command()
def ingest(
    codebase_path: str = typer.Argument(".", help="Path to the codebase to ingest"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to the configuration file"
    ),
    exclude: List[str] = typer.Option(
        [], "--exclude", "-e", help="Patterns to exclude from ingestion"
    ),
    include: List[str] = typer.Option(
        [], "--include", "-i", help="Directories to specifically include"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force reindexing of the codebase"
    ),
) -> None:
    """Ingest and index a codebase for querying and documentation.

    This command processes your codebase files, generates embeddings, and creates
    searchable indexes. For OpenAI embeddings, token usage and costs are tracked
    and displayed during the process.
    """
    # Show initial information
    abs_codebase_path = Path(codebase_path).resolve()

    user_config = load_or_init_config(config_path)

    # Update config with command-line overrides
    if exclude or include:
        # Make a copy of the user_config for modification
        updated_config = user_config

        # Create or get the ingestion configuration
        if updated_config.ingestion is None:
            from docstra.core.config.settings import IngestionConfig

            # Create with default values and override as needed
            updated_config.ingestion = IngestionConfig(
                include_dirs=None, exclude_patterns=[]
            )

        # Update exclude patterns
        if exclude:
            updated_config.ingestion.exclude_patterns = exclude

        # Update include dirs
        if include:
            updated_config.ingestion.include_dirs = include

        # Use the updated config
        user_config = updated_config

    # Create a clean header for ingestion
    console.print(
        Panel(
            f"[{Colors.BOLD}]🚀 Processing Codebase[/]\n"
            f"📁 [{Colors.DIM}]{abs_codebase_path}[/]\n"
            f"🤖 [{Colors.DIM}]{user_config.model.provider} - {user_config.model.model_name}[/]\n"
            f"🔗 [{Colors.DIM}]{user_config.embedding.provider} - {user_config.embedding.model_name}[/]"
            + (
                f"\n⚠️  [{Colors.WARNING}]OpenAI embeddings - API costs will apply[/]"
                if user_config.embedding.provider.lower() == "openai"
                else ""
            )
            + (
                f"\n🔄 [{Colors.WARNING}]Force mode - will reindex existing data[/]"
                if force
                else ""
            ),
            style=Colors.INFO_BOLD,
            expand=False,
        )
    )

    # Create ingestion service for this operation
    ingestion_service, _, _, _ = create_services_for_config(user_config)

    # Run ingestion using the service
    success = ingestion_service.ingest_codebase(
        codebase_path=codebase_path, user_config=user_config, force=force
    )

    if not success:
        console.print(f"[{Colors.ERROR_BOLD}]Ingestion failed.[/]")
        raise typer.Exit(code=1)

    # Show next steps with emojis and cleaner formatting
    console.print("\n" + "─" * 50)
    console.print(f"[{Colors.SUCCESS_BOLD}]🎉 Ingestion Complete![/]")
    console.print("─" * 50)
    console.print(f"[{Colors.BOLD}]🔍 Try these commands:[/]")
    console.print(
        f'   • [{Colors.HIGHLIGHT}]docstra query[/] "your question" - Ask questions about your code'
    )
    console.print(
        f"   • [{Colors.HIGHLIGHT}]docstra chat[/] - Start an interactive chat session"
    )
    console.print(
        f"   • [{Colors.HIGHLIGHT}]docstra generate[/] - Generate comprehensive documentation"
    )
    console.print("─" * 50)


@app.command()
def index(
    codebase_path: str = typer.Argument(".", help="Path to the codebase to index"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to the configuration file"
    ),
    export: Optional[str] = typer.Option(
        None, "--export", help="Write the core index manifest JSON to this path"
    ),
) -> None:
    """Build the deterministic core index (files, symbols, imports, edges).

    No embeddings are generated, so no model or API access is needed.
    Run 'docstra ingest' afterwards to add embeddings for semantic retrieval.
    """
    abs_codebase_path = Path(codebase_path).resolve()
    user_config = load_or_init_config(config_path)

    console.print(
        Panel(
            f"[{Colors.BOLD}]🗂️  Building Core Index[/]\n"
            f"📁 [{Colors.DIM}]{abs_codebase_path}[/]",
            style=Colors.INFO_BOLD,
            expand=False,
        )
    )

    ingestion_service = IngestionService(console=console)
    success = ingestion_service.build_core_index(
        codebase_path=str(abs_codebase_path),
        user_config=user_config,
        export_path=export,
    )

    if not success:
        console.print(f"[{Colors.ERROR_BOLD}]Indexing failed.[/]")
        raise typer.Exit(code=1)


def format_file_link(abs_path: str, start_line, end_line) -> str:
    """Format a link with line numbers for Rich clickable links."""
    file_url = f"{quote(abs_path)}"
    output = f"{file_url}"
    if start_line != "?" and end_line != "?":
        output += f":{start_line}"
        if start_line == end_line:
            output += f" (#L{start_line})"
        elif start_line < end_line:
            output += f" (#L{start_line}-L{end_line})"

    return f"[{Colors.HIGHLIGHT}]{output}[/{Colors.HIGHLIGHT}]"


def postprocess_llm_output_with_links(answer: str, sources: list) -> str:
    """
    Postprocess the LLM output to replace file/method/class references with clickable Rich links if possible.
    This scans for file references in the answer and replaces them with [link=...]...[/link] using the sources' metadata.
    """
    # Build a mapping from file/method/class names to file links
    file_links = {}
    for source in sources:
        meta = source.get("metadata", {})
        filepath = meta.get("document_id", "")
        start_line = meta.get("start_line", "?")
        end_line = meta.get("end_line", "?")
        try:
            abs_path = str(Path(filepath).resolve())
        except Exception:
            abs_path = filepath
        display = abs_path
        if start_line != "?" and end_line != "?":
            if start_line == end_line:
                display = f"{abs_path}:{start_line}"
            else:
                display = f"{abs_path}:{start_line}-{end_line}"
        file_url = format_file_link(abs_path, start_line, end_line)
        # Add by filename
        file_links[os.path.basename(abs_path)] = (file_url, display)
        # Add by full path
        file_links[abs_path] = (file_url, display)
        # Add by method/class name if available
        for key in ("symbol", "function", "class"):
            if key in meta:
                file_links[meta[key]] = (file_url, display)

    # Regex to find file references (filenames, file.py:123, etc.)
    file_ref_pattern = re.compile(r"([\w\-/]+\.py(?::\d+(?:-\d+)?)?)")

    def replacer(match):
        ref = match.group(1)
        # Try to find a link for this reference
        for key, (url, display) in file_links.items():
            if ref == key or ref in display or ref in key:
                return f"[link={url}][cyan]{ref}[/cyan][/link]"
        return ref  # No link found

    # Replace file references in the answer
    processed = file_ref_pattern.sub(replacer, answer)
    return processed


def _get_persist_paths(
    user_config: UserConfig, abs_codebase_path: Path
) -> tuple[Path, Path, Path]:
    persist_directory_name = user_config.storage.persist_directory
    if not Path(persist_directory_name).is_absolute():
        effective_persist_dir = abs_codebase_path / persist_directory_name
    else:
        effective_persist_dir = Path(persist_directory_name)

    effective_persist_dir = effective_persist_dir.resolve()
    chroma_path = effective_persist_dir / "chroma"
    index_path = effective_persist_dir / "index"
    return effective_persist_dir, chroma_path, index_path


def _create_retrieval_eval_runner(
    user_config: UserConfig, abs_codebase_path: Path
) -> Callable[[str, int], List[Dict[str, Any]]]:
    effective_persist_dir, chroma_path, index_path = _get_persist_paths(
        user_config, abs_codebase_path
    )
    core_index_path = index_path / CORE_INDEX_FILENAME
    chroma_check_file = chroma_path / "chroma.sqlite3"
    legacy_index_artifacts = CodebaseIndex.legacy_artifacts_in(index_path)
    legacy_repo_map = index_path.parent / "repo_map.json"

    if not core_index_path.exists() or not chroma_check_file.exists():
        migration_hint = ""
        if legacy_index_artifacts or legacy_repo_map.exists():
            migration_hint = (
                " Legacy index artifacts were found. Rerun 'docstra ingest' "
                "to rebuild the index in the new format."
            )
        raise FileNotFoundError(
            f"Codebase at {abs_codebase_path} is not fully initialized for "
            f"retrieval evaluation. ChromaDB path: {chroma_path} "
            f"(check file: {chroma_check_file}, exists: "
            f"{chroma_check_file.exists()}), core index path: {core_index_path} "
            f"(exists: {core_index_path.exists()}). Run 'docstra init' and "
            f"'docstra ingest' first.{migration_hint}"
        )

    embedding_generator = EmbeddingFactory.create_embedding_generator(
        embedding_type=user_config.embedding.provider,
        model_name=user_config.embedding.model_name,
        api_key=user_config.embedding.api_key or user_config.model.api_key,
        api_base=user_config.model.api_base,
    )
    storage = ChromaDBStorage(persist_directory=str(chroma_path))
    base_retriever = ChromaRetriever(
        storage,
        embedding_generator,
        codebase_root=str(abs_codebase_path),
    )
    code_indexer = CodebaseIndexer(
        index_directory=str(index_path),
        codebase_root=str(abs_codebase_path),
    )
    code_index = code_indexer.get_index()

    if code_index:
        fts_storage = FtsStorage(str(effective_persist_dir / "index.db"))
        fts_retriever = FtsRetriever(fts_storage)
        fusion_retriever = FusionRetriever(
            dense=base_retriever,
            fts=fts_retriever,
            code_index=code_index,
            rrf_k=user_config.retrieval.rrf_k,
            fts_chunks_top_k=user_config.retrieval.fts_chunks_top_k,
            fts_symbols_top_k=user_config.retrieval.fts_symbols_top_k,
        )

        def retrieve(question: str, top_k: int) -> List[Dict[str, Any]]:
            return fusion_retriever.retrieve(question, n_results=top_k)

        return retrieve

    def retrieve(question: str, top_k: int) -> List[Dict[str, Any]]:
        return base_retriever.retrieve_chunks(question, n_results=top_k)

    return retrieve


def _print_retrieval_eval_summary(summary: RetrievalEvalSummary) -> None:
    table = Table(
        title=f"Retrieval Eval Results (recall@{summary.top_k})",
        show_header=True,
        header_style=Colors.HIGHLIGHT_BOLD,
    )
    table.add_column("#", justify="right")
    table.add_column("Question", overflow="fold")
    table.add_column("Expected", overflow="fold")
    table.add_column("Match", overflow="fold")
    table.add_column("Rank", justify="right")
    table.add_column("Result")

    for index, result in enumerate(summary.results, start=1):
        table.add_row(
            str(index),
            result.case.question,
            ", ".join(result.case.expected_files),
            result.matched_file or "-",
            str(result.rank) if result.rank is not None else "-",
            "pass" if result.passed else "fail",
            style=Colors.SUCCESS if result.passed else Colors.ERROR,
        )

    console.print(table)
    console.print(
        f"[{Colors.BOLD}]Recall@{summary.top_k}:[/] "
        f"{summary.passed_count}/{summary.total} "
        f"({summary.recall_at_k:.1%})"
    )


@app.command("eval-retrieval")
def eval_retrieval(
    codebase_path: str = typer.Option(
        ".", "--codebase", "-C", help="Path to the codebase"
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to the configuration file"
    ),
    top_k: int = typer.Option(
        10, "--top-k", min=1, help="Number of unique source files to evaluate"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print machine-readable JSON output"
    ),
) -> None:
    """Evaluate retrieval against Docstra's built-in source-file checks."""
    user_config = load_or_init_config(config_path)
    abs_codebase_path = Path(codebase_path).resolve()

    try:
        retrieve = _create_retrieval_eval_runner(user_config, abs_codebase_path)
        candidate_k = max(
            top_k * RETRIEVAL_EVAL_CANDIDATE_MULTIPLIER,
            RETRIEVAL_EVAL_MIN_CANDIDATES,
        )
        summary = evaluate_retrieval_cases(
            cases=DEFAULT_RETRIEVAL_EVAL_CASES,
            retrieve=retrieve,
            top_k=top_k,
            candidate_k=candidate_k,
            codebase_path=abs_codebase_path,
        )
    except FileNotFoundError as exc:
        console.print(f"[{Colors.ERROR_BOLD}]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"[{Colors.ERROR_BOLD}]Error during retrieval eval:[/] {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        console.print(json.dumps(summary.to_dict(), indent=2))
        return

    _print_retrieval_eval_summary(summary)


# Add query command - refactored from ask
@app.command()
def query(
    question: str = typer.Argument(..., help="Question about the codebase"),
    codebase_path: str = typer.Option(
        ".", "--codebase", "-C", help="Path to the codebase"
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to the configuration file"
    ),
    n_results: int = typer.Option(
        5, "--results", "-n", help="Number of results to retrieve"
    ),
    query_type: Optional[str] = typer.Option(
        None, "--type", help="Query type: architectural, implementation, usage"
    ),
    symbols_only: bool = typer.Option(
        False, "--symbols-only", help="Search symbols only"
    ),
    with_context: bool = typer.Option(
        True, "--context/--no-context", help="Include repository context in answer"
    ),
    detailed: bool = typer.Option(
        False, "--detailed", help="Show detailed analysis and relationships"
    ),
    context_mode: Optional[str] = typer.Option(
        None, "--context-mode", help="Context mode: compact, balanced, detailed"
    ),
) -> None:
    """Ask a question about the codebase and get a precise answer with enhanced context."""
    # Get configuration
    user_config = load_or_init_config(config_path)

    # Apply context mode override if provided
    if context_mode:
        if context_mode not in ["compact", "balanced", "detailed"]:
            console.print(
                f"[{Colors.ERROR_BOLD}]Error: Invalid context mode '{context_mode}'. Must be: compact, balanced, detailed[/]"
            )
            raise typer.Exit(code=1)
        user_config.model.context_mode = context_mode

    # Create query service for this operation
    _, query_service_with_config, _, _ = create_services_for_config(user_config)

    # Validate LLM connection if using Ollama
    if user_config.model.provider == ModelProvider.OLLAMA:
        from docstra.core.llm.ollama import OllamaClient

        if isinstance(query_service_with_config.llm_client, OllamaClient):
            is_connected, message = (
                query_service_with_config.llm_client.validate_connection()
            )
            if not is_connected:
                console.print(f"[{Colors.ERROR_BOLD}]Error:[/] {message}")
                raise typer.Exit(code=1)

    # Enhanced query with repository context
    enhanced_question = question

    # Add query type context if specified
    if query_type:
        if query_type.lower() == "architectural":
            enhanced_question = f"From an architectural perspective: {question}"
        elif query_type.lower() == "implementation":
            enhanced_question = f"From an implementation detail perspective: {question}"
        elif query_type.lower() == "usage":
            enhanced_question = f"From a usage and examples perspective: {question}"

    # Modify number of results based on symbols_only flag
    search_results = n_results
    if symbols_only:
        search_results = min(
            n_results * 2, 15
        )  # Get more results to filter for symbols

    # Call answer_question with the right parameters
    answer, sources = query_service_with_config.answer_question(
        question=enhanced_question,
        codebase_path_str=codebase_path,
        n_results=search_results,
    )

    # Postprocess the answer to add clickable links
    processed_answer = postprocess_llm_output_with_links(str(answer), sources)

    # Display the answer
    console.print(Panel(Markdown(processed_answer), title="Answer"))

    # Show sources if available
    if sources:
        console.print(f"\n[{Colors.BOLD}]Sources:[/]")
        for i, source in enumerate(sources[:5]):
            meta = source.get("metadata", {})
            filepath = meta.get("document_id", "Unknown")
            start_line = meta.get("start_line", "?")
            end_line = meta.get("end_line", "?")
            try:
                abs_path = str(Path(filepath).resolve())
            except Exception:
                abs_path = filepath
            link_str = format_file_link(abs_path, start_line, end_line)
            console.print(f"[{Colors.BOLD}]{i + 1}.[/] {link_str}")

    # Show detailed analysis if requested
    if detailed and with_context:
        try:
            from docstra.core.services.repository_explorer_service import (
                RepositoryExplorerService,
            )

            explorer_service = RepositoryExplorerService(user_config, console)

            # Show repository context for the question
            console.print(f"\n[{Colors.BOLD}]Repository Context Analysis:[/]")

            # If sources mention specific files, show their relationships
            mentioned_files = []
            for source in sources[:3]:  # Analyze top 3 sources
                meta = source.get("metadata", {})
                filepath = meta.get("document_id")
                if filepath and os.path.exists(filepath):
                    mentioned_files.append(filepath)

            if mentioned_files:
                console.print(f"\n[{Colors.INFO_BOLD}]Key Files in Answer:[/]")

                for file_path in mentioned_files:
                    try:
                        relationships = explorer_service.get_file_relationships(
                            file_path
                        )

                        # Show brief file context
                        file_table = Table(
                            title=f"Context for {os.path.basename(file_path)}",
                            show_header=True,
                            header_style="bold cyan",
                        )
                        file_table.add_column("Aspect", style="cyan")
                        file_table.add_column("Details", style="white")

                        complexity_info = relationships.get("complexity_info", {})
                        arch_info = relationships.get("architectural_info", {})

                        file_table.add_row(
                            "Module Type", complexity_info.get("module_type", "Unknown")
                        )
                        file_table.add_row(
                            "Dependencies",
                            str(len(relationships.get("dependencies", []))),
                        )
                        file_table.add_row(
                            "Dependents", str(len(relationships.get("dependents", [])))
                        )
                        file_table.add_row(
                            "Core Module",
                            "Yes" if arch_info.get("is_core_module", False) else "No",
                        )

                        console.print(file_table)

                    except Exception as e:
                        console.print(
                            f"[{Colors.DIM}]Could not analyze {os.path.basename(file_path)}: {e}[/]"
                        )

        except Exception as e:
            console.print(f"[{Colors.WARNING}]Could not load detailed context: {e}[/]")

    # Add query suggestions based on the current question
    if with_context:
        console.print(f"\n[{Colors.DIM}]💡 Try these related queries:[/]")
        if query_type != "architectural":
            console.print(
                f'[{Colors.DIM}]• docstra query "{question}" --type architectural[/]'
            )
        if query_type != "implementation":
            console.print(
                f'[{Colors.DIM}]• docstra query "{question}" --type implementation[/]'
            )
        if not symbols_only:
            console.print(
                f'[{Colors.DIM}]• docstra query "{question}" --symbols-only[/]'
            )

    # Display enhanced token usage statistics
    llm_tracker = get_llm_tracker()
    if llm_tracker and llm_tracker.session_stats:
        console.print(f"\n[{Colors.DIM}]Query Summary:[/]")

        # Get the last usage from session stats
        usage = llm_tracker.session_stats[-1]

        # Get model and context window info
        model_name = user_config.model.model_name
        provider = user_config.model.provider.value
        context_window = user_config.model.context_window

        # If no context window configured, get estimated from token counter
        if not context_window:
            from docstra.core.utils.token_counter import get_token_counter

            token_counter = get_token_counter(model_name, provider)
            context_window = token_counter.estimate_max_context()

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_used = input_tokens + output_tokens
        remaining_context = context_window - total_used

        console.print(
            f"[{Colors.DIM}]Model: {model_name} ({provider}) | Context: {context_window:,} tokens[/]"
        )
        console.print(
            f"[{Colors.DIM}]Input: {input_tokens:,} tokens | Output: {output_tokens:,} tokens | Used: {total_used:,} tokens[/]"
        )
        console.print(
            f"[{Colors.DIM}]Remaining context: {remaining_context:,} tokens ({(remaining_context / context_window) * 100:.1f}%)[/]"
        )

        if "cost_usd" in usage and usage.get("cost_usd", 0) > 0:
            console.print(
                f"[{Colors.DIM}]Estimated cost: ${usage.get('cost_usd', 0):.5f}[/]"
            )


# Add chat command - new functionality per refactoring plan
@app.command()
def chat(
    codebase_path: str = typer.Argument(
        ".", help="Path to the codebase for the chat session"
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to the configuration file"
    ),
    session_id: Optional[str] = typer.Option(
        None, "--session-id", "-s", help="Resume an existing chat session by ID"
    ),
    new_session: bool = typer.Option(
        False, "--new", "-n", help="Start a new session even if session-id is provided"
    ),
    list_sessions: bool = typer.Option(
        False, "--list", "-l", help="List available chat sessions"
    ),
    delete_session: Optional[str] = typer.Option(
        None, "--delete", "-d", help="Delete a chat session by ID"
    ),
    context_mode: Optional[str] = typer.Option(
        None, "--context-mode", help="Context mode: compact, balanced, detailed"
    ),
) -> None:
    """Start an interactive chat session with the codebase assistant."""
    # Get configuration
    user_config = load_or_init_config(config_path)

    # Apply context mode override if provided
    if context_mode:
        if context_mode not in ["compact", "balanced", "detailed"]:
            console.print(
                f"[{Colors.ERROR_BOLD}]Error: Invalid context mode '{context_mode}'. Must be: compact, balanced, detailed[/]"
            )
            raise typer.Exit(code=1)
        user_config.model.context_mode = context_mode

    # Create chat service for this operation
    _, _, chat_service, _ = create_services_for_config(user_config)

    # Validate LLM connection if using Ollama
    if user_config.model.provider == ModelProvider.OLLAMA:
        from docstra.core.llm.ollama import OllamaClient

        if hasattr(chat_service, "llm_client") and isinstance(
            chat_service.llm_client, OllamaClient
        ):
            is_connected, message = chat_service.llm_client.validate_connection()
            if not is_connected:
                console.print(f"[{Colors.ERROR_BOLD}]Error:[/] {message}")
                raise typer.Exit(code=1)

    # Handle session management options
    if list_sessions:
        sessions = chat_service.list_sessions()
        if not sessions:
            console.print(f"[{Colors.WARNING}]No chat sessions found.[/]")
        return

        console.print(f"[{Colors.BOLD}]Available chat sessions:[/]")
        for i, session in enumerate(sessions):
            console.print(
                f"[{Colors.BOLD}]{i + 1}.[/] [{Colors.HIGHLIGHT}]{session['name']}[/] "
                f"(ID: {session['id']}, Last accessed: {session['last_accessed_at']})"
            )
        return

    if delete_session:
        success = chat_service.delete_session(delete_session)
        if success:
            console.print(
                f"[{Colors.SUCCESS}]Session {delete_session} deleted successfully.[/]"
            )
        else:
            console.print(
                f"[{Colors.ERROR}]Failed to delete session {delete_session}.[/]"
            )
        return

    # Start or resume a session
    chat_service.start_new_session(codebase_path)

    # Interactive chat loop
    console.print(
        f"[{Colors.BOLD}]Chat session started. Type '/exit' or '/quit' to end the session.[/]"
    )
    console.print(f"[{Colors.BOLD}]Type '/help' for available commands.[/]")
    console.print(
        f"[{Colors.DIM}]💡 Commands use '/' prefix (e.g., /sessions, /stats). Regular questions don't need any prefix.[/]"
    )

    while True:
        try:
            # Get user input
            user_input = input("\n[You]: ")

            # Check for exit/quit command
            if user_input.lower() in ["/exit", "/quit"]:
                console.print(f"[{Colors.BOLD}]Ending chat session.[/]")
                break

            # Check for unknown commands first (before processing known commands)
            if user_input.startswith("/") and len(user_input) > 1:
                # Extract the command part (before any space)
                command_part = user_input.split()[0].lower()

                # List of valid commands
                valid_commands = {
                    "/help",
                    "/clear",
                    "/stats",
                    "/sessions",
                    "/switch",
                    "/new",
                    "/delete",
                    "/info",
                    "/history",
                    "/export",
                }

                # Check if it's an unknown command
                if command_part not in valid_commands and not any(
                    command_part.startswith(cmd) for cmd in ["/switch", "/delete"]
                ):
                    console.print(
                        f"\n[{Colors.ERROR_BOLD}]❌ Unknown command:[/] {command_part}"
                    )
                    console.print(
                        f"[{Colors.DIM}]Type [{Colors.HIGHLIGHT}]/help[/] to see all available commands[/]"
                    )
                    console.print(
                        f"[{Colors.DIM}]💡 Commands use '/' prefix. Regular questions don't need any prefix.[/]"
                    )
                    continue

            # Check for help command
            if user_input.lower() in ["help", "/help"]:
                console.print(f"\n[{Colors.BOLD}]Available commands:[/]")
                console.print(
                    f"  [{Colors.HIGHLIGHT}]exit/quit[/] - End the chat session"
                )
                console.print(
                    f"  [{Colors.HIGHLIGHT}]/help[/] - Show this help message"
                )
                console.print(f"  [{Colors.HIGHLIGHT}]/clear[/] - Clear the screen")
                console.print(
                    f"  [{Colors.HIGHLIGHT}]/stats[/] - Show token usage statistics"
                )
                console.print(f"\n[{Colors.BOLD}]Session Management:[/]")
                console.print(
                    f"  [{Colors.HIGHLIGHT}]/sessions[/] - List all chat sessions"
                )
                console.print(
                    f"  [{Colors.HIGHLIGHT}]/switch <session_id>[/] - Switch to another session"
                )
                console.print(
                    f"  [{Colors.HIGHLIGHT}]/new[/] - Start a new chat session"
                )
                console.print(
                    f"  [{Colors.HIGHLIGHT}]/delete <session_id>[/] - Delete a session"
                )
                console.print(
                    f"  [{Colors.HIGHLIGHT}]/info[/] - Show current session details"
                )
                console.print(
                    f"  [{Colors.HIGHLIGHT}]/history[/] - Show recent conversation history"
                )
                console.print(
                    f"  [{Colors.HIGHLIGHT}]/export[/] - Export current conversation to file"
                )
                console.print(
                    f"\n[{Colors.DIM}]💡 Tip: You can use partial session IDs (e.g., '/switch abc123')[/]"
                )
                continue

            # Check for clear command
            if user_input.lower() == "/clear":
                subprocess.run(
                    ["cls" if os.name == "nt" else "clear"],
                    check=False,
                    shell=os.name == "nt",
                )
                continue

            # Check for stats command
            if user_input.lower() == "/stats":
                console.print(f"\n[{Colors.BOLD}]Chat Statistics:[/]")

                # Show detailed last interaction stats
                if chat_service.last_interaction:
                    interaction = chat_service.last_interaction
                    console.print(
                        f"\n[{Colors.BOLD}]Last Question:[/] {interaction['user_query'][:60]}{'...' if len(interaction['user_query']) > 60 else ''}"
                    )
                    console.print(
                        f"[{Colors.BOLD}]Model:[/] {interaction['model_name']} ({interaction['provider']})"
                    )

                    # Get context window info
                    context_window = user_config.model.context_window
                    if not context_window:
                        from docstra.core.utils.token_counter import get_token_counter

                        token_counter = get_token_counter(
                            interaction["model_name"], interaction["provider"]
                        )
                        context_window = token_counter.estimate_max_context()

                    total_tokens = interaction["total_tokens"]
                    usage_percent = (total_tokens / context_window) * 100
                    remaining = context_window - total_tokens

                    console.print(
                        f"[{Colors.BOLD}]Context Window:[/] {context_window:,} tokens"
                    )
                    console.print(f"[{Colors.BOLD}]Token Breakdown:[/]")
                    console.print(
                        f"  • Conversation history: {interaction['conversation_tokens']:,} tokens"
                    )
                    console.print(
                        f"  • Retrieved context: {interaction['context_tokens']:,} tokens"
                    )
                    console.print(
                        f"  • Response generated: {interaction['response_tokens']:,} tokens"
                    )
                    console.print(
                        f"  • Total used: {total_tokens:,} tokens ({usage_percent:.1f}%)"
                    )
                    console.print(
                        f"  • Remaining: {remaining:,} tokens ({(remaining / context_window) * 100:.1f}%)"
                    )
                    console.print(
                        f"[{Colors.BOLD}]Context Mode:[/] {interaction['context_mode']} | Sources: {interaction['sources_count']}"
                    )
                else:
                    console.print("No interaction data available yet.")

                # Show session summary
                llm_tracker = get_llm_tracker()
                if llm_tracker:
                    session_summary = llm_tracker.get_session_summary()
                    if "session_summary" in session_summary:
                        console.print(f"\n[{Colors.BOLD}]Session Summary:[/]")
                        session_stats = session_summary.get("session_summary", {})
                        console.print(
                            f"Total requests: {session_stats.get('total_requests', 0)}"
                        )
                        console.print(
                            f"Total input tokens: {session_stats.get('total_input_tokens', 0):,}"
                        )
                        console.print(
                            f"Total output tokens: {session_stats.get('total_output_tokens', 0):,}"
                        )
                        if session_stats.get("total_cost", 0) > 0:
                            console.print(
                                f"Total cost: ${session_stats.get('total_cost', 0):.5f}"
                            )
                        console.print(
                            f"Average duration: {session_stats.get('total_duration_ms', 0) / max(session_stats.get('total_requests', 1), 1):.0f} ms/request"
                        )

                continue

            # Session management commands
            if user_input.lower() == "/sessions":
                sessions = chat_service.list_sessions()
                if not sessions:
                    console.print(f"[{Colors.WARNING}]No chat sessions found.[/]")
                else:
                    console.print(f"\n[{Colors.BOLD}]Available Chat Sessions:[/]")
                    for i, session in enumerate(sessions):
                        session_marker = (
                            "📍"
                            if session["id"] == chat_service.current_session_id
                            else "  "
                        )
                        console.print(
                            f"{session_marker} [{Colors.HIGHLIGHT}]{session['id'][:8]}[/] - {session['name']} "
                            f"[{Colors.DIM}](last used: {session['last_accessed_at'][:16]})[/]"
                        )
                    console.print(
                        f"\n[{Colors.DIM}]Use '/switch <session_id>' to change sessions[/]"
                    )
                continue

            if user_input.lower().startswith("/switch "):
                session_id = user_input[8:].strip()
                if not session_id:
                    console.print(
                        f"[{Colors.ERROR}]Please provide a session ID. Use '/sessions' to see available sessions.[/]"
                    )
                    continue

                # Try to find session by partial ID match
                sessions = chat_service.list_sessions()
                matching_session = None
                for session in sessions:
                    if session["id"].startswith(session_id):
                        matching_session = session
                        break

                if not matching_session:
                    console.print(
                        f"[{Colors.ERROR}]Session '{session_id}' not found. Use '/sessions' to see available sessions.[/]"
                    )
                    continue

                if chat_service.load_session(matching_session["id"], codebase_path):
                    console.print(
                        f"[{Colors.SUCCESS}]Switched to session: {matching_session['name']}[/]"
                    )
                else:
                    console.print(
                        f"[{Colors.ERROR}]Failed to switch to session '{session_id}'[/]"
                    )
                continue

            if user_input.lower() == "/new":
                old_session_name = None
                if chat_service.current_session_id:
                    sessions = chat_service.list_sessions()
                    for session in sessions:
                        if session["id"] == chat_service.current_session_id:
                            old_session_name = session["name"]
                            break

                chat_service.start_new_session(codebase_path)
                console.print(f"[{Colors.SUCCESS}]Started new chat session[/]")
                if old_session_name:
                    console.print(
                        f"[{Colors.DIM}]Previous session '{old_session_name}' is still available[/]"
                    )
                continue

            if user_input.lower().startswith("/delete "):
                session_id = user_input[8:].strip()
                if not session_id:
                    console.print(
                        f"[{Colors.ERROR}]Please provide a session ID. Use '/sessions' to see available sessions.[/]"
                    )
                    continue

                # Prevent deleting current session
                if (
                    chat_service.current_session_id
                    and chat_service.current_session_id.startswith(session_id)
                ):
                    console.print(
                        f"[{Colors.ERROR}]Cannot delete the current active session. Switch to another session first.[/]"
                    )
                    continue

                # Try to find session by partial ID match
                sessions = chat_service.list_sessions()
                matching_session = None
                for session in sessions:
                    if session["id"].startswith(session_id):
                        matching_session = session
                        break

                if not matching_session:
                    console.print(
                        f"[{Colors.ERROR}]Session '{session_id}' not found. Use '/sessions' to see available sessions.[/]"
                    )
                    continue

                if chat_service.delete_session(matching_session["id"]):
                    console.print(
                        f"[{Colors.SUCCESS}]Deleted session: {matching_session['name']}[/]"
                    )
                else:
                    console.print(
                        f"[{Colors.ERROR}]Failed to delete session '{session_id}'[/]"
                    )
                continue

            if user_input.lower() == "/info":
                if not chat_service.current_session_id:
                    console.print(f"[{Colors.WARNING}]No active session[/]")
                    continue

                sessions = chat_service.list_sessions()
                current_session = None
                for session in sessions:
                    if session["id"] == chat_service.current_session_id:
                        current_session = session
                        break

                if current_session:
                    console.print(f"\n[{Colors.BOLD}]Current Session Info:[/]")
                    console.print(f"[{Colors.BOLD}]Name:[/] {current_session['name']}")
                    console.print(f"[{Colors.BOLD}]ID:[/] {current_session['id']}")
                    console.print(
                        f"[{Colors.BOLD}]Created:[/] {current_session['created_at']}"
                    )
                    console.print(
                        f"[{Colors.BOLD}]Last Used:[/] {current_session['last_accessed_at']}"
                    )
                    console.print(
                        f"[{Colors.BOLD}]Codebase:[/] {current_session['codebase_path']}"
                    )
                    console.print(
                        f"[{Colors.BOLD}]Messages:[/] {len(chat_service.current_chat_history)} in conversation"
                    )
                else:
                    console.print(
                        f"[{Colors.WARNING}]Session information not available[/]"
                    )
                continue

            if user_input.lower() == "/history":
                if not chat_service.current_chat_history:
                    console.print(
                        f"[{Colors.WARNING}]No conversation history in current session[/]"
                    )
                    continue

                console.print(f"\n[{Colors.BOLD}]Recent Conversation History:[/]")
                # Show last 10 messages
                recent_messages = chat_service.current_chat_history[-10:]
                for i, msg in enumerate(recent_messages):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")

                    # Truncate long messages
                    if len(content) > 100:
                        content = content[:100] + "..."

                    role_color = Colors.HIGHLIGHT if role == "user" else Colors.SUCCESS
                    console.print(f"[{role_color}]{role.capitalize()}:[/] {content}")

                if len(chat_service.current_chat_history) > 10:
                    console.print(
                        f"[{Colors.DIM}]... and {len(chat_service.current_chat_history) - 10} older messages[/]"
                    )
                continue

            if user_input.lower() == "/export":
                if not chat_service.current_chat_history:
                    console.print(
                        f"[{Colors.WARNING}]No conversation history to export[/]"
                    )
                    continue

                # Generate filename with timestamp
                import datetime

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"docstra_chat_{timestamp}.md"

                try:
                    with open(filename, "w", encoding="utf-8") as f:
                        # Get session info
                        sessions = chat_service.list_sessions()
                        current_session = None
                        for session in sessions:
                            if session["id"] == chat_service.current_session_id:
                                current_session = session
                                break

                        # Write header
                        f.write("# Docstra Chat Export\n\n")
                        if current_session:
                            f.write(f"**Session:** {current_session['name']}\n")
                            f.write(f"**Created:** {current_session['created_at']}\n")
                            f.write(
                                f"**Codebase:** {current_session['codebase_path']}\n"
                            )
                        f.write(
                            f"**Exported:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        )
                        f.write("---\n\n")

                        # Write conversation
                        for msg in chat_service.current_chat_history:
                            role = msg.get("role", "unknown").capitalize()
                            content = msg.get("content", "")
                            f.write(f"## {role}\n\n{content}\n\n")

                    console.print(
                        f"[{Colors.SUCCESS}]Conversation exported to: {filename}[/]"
                    )
                except Exception as e:
                    console.print(
                        f"[{Colors.ERROR}]Failed to export conversation: {e}[/]"
                    )
                continue

            # Get response from the chat service
            with console.status(f"[{Colors.INFO}]Thinking...", spinner="dots"):
                response = chat_service.get_response(user_input)

            # Display the response
            console.print(f"\n[Assistant]: {response}")

            # Simple one-line context summary
            console.print(
                f"[{Colors.DIM}]Used {chat_service.get_last_usage_summary()}[/]"
            )

        except KeyboardInterrupt:
            console.print(f"\n[{Colors.BOLD}]Chat session interrupted.[/]")
            break
        except EOFError:
            console.print(f"\n[{Colors.BOLD}]Chat session ended.[/]")
            break
        except Exception as e:
            console.print(f"\n[{Colors.ERROR_BOLD}]Error in chat: {e}[/]")


@app.command()
def detect(
    codebase_path: str = typer.Argument(".", help="Path to the codebase to analyze"),
    show_patterns: bool = typer.Option(
        False, "--show-patterns", help="Show generated ignore patterns"
    ),
) -> None:
    """Detect languages and frameworks in a codebase and show recommended ignore patterns."""
    console.print(Panel("Codebase Language & Framework Detection", expand=False))

    # Initialize detector
    detector = LanguageDetector(codebase_path)

    # Get detection summary
    with console.status(f"[{Colors.INFO}]Analyzing codebase...", spinner="dots"):
        summary = detector.get_detection_summary()

    # Display results
    console.print(
        f"\n[{Colors.BOLD}]Codebase Analysis Results for:[/] {Path(codebase_path).resolve()}"
    )
    console.print(
        f"[{Colors.BOLD}]Primary Language:[/] [{Colors.SUCCESS}]{summary['primary_language']}[/]"
    )
    console.print(
        f"[{Colors.BOLD}]Codebase Type:[/] [{Colors.SUCCESS}]{summary['codebase_type']}[/]"
    )

    # Show language breakdown
    if summary["languages"]:
        console.print(f"\n[{Colors.BOLD}]Languages Detected:[/]")
        for language, count in sorted(
            summary["languages"].items(), key=lambda x: x[1], reverse=True
        ):
            console.print(f"  • [{Colors.HIGHLIGHT}]{language}[/]: {count} files")

    # Show frameworks
    if summary["frameworks"]:
        console.print(f"\n[{Colors.BOLD}]Frameworks/Tools Detected:[/]")
        for framework in sorted(summary["frameworks"]):
            console.print(f"  • [{Colors.WARNING}]{framework}[/]")

    # Show pattern count
    console.print(
        f"\n[{Colors.BOLD}]Recommended Ignore Patterns:[/] {summary['total_patterns']} patterns"
    )

    # Show patterns if requested
    if show_patterns:
        console.print(f"\n[{Colors.BOLD}]Generated Ignore Patterns:[/]")
        for pattern in summary["ignore_patterns"]:
            console.print(f"  {pattern}")
    else:
        console.print(f"[{Colors.DIM}]Use --show-patterns to see the full list[/]")

    # Show recommendations
    console.print(f"\n[{Colors.BOLD}]Recommendations:[/]")
    if summary["total_patterns"] < 20:
        console.print(
            f"  • [{Colors.SUCCESS}]Lightweight pattern set - good for performance[/]"
        )
    elif summary["total_patterns"] > 50:
        console.print(
            f"  • [{Colors.WARNING}]Large pattern set - consider reviewing for optimization[/]"
        )
    else:
        console.print(
            f"  • [{Colors.SUCCESS}]Balanced pattern set for your project type[/]"
        )

    if summary["codebase_type"] == "web_frontend":
        console.print(
            f"  • [{Colors.INFO}]Consider adding framework-specific build directories to .gitignore[/]"
        )
    elif summary["codebase_type"] == "python":
        console.print(
            f"  • [{Colors.INFO}]Virtual environment directories are automatically excluded[/]"
        )
    elif summary["codebase_type"] == "mobile":
        console.print(
            f"  • [{Colors.INFO}]Platform-specific build artifacts are excluded[/]"
        )


@app.command()
def usage(
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to the configuration file"
    ),
    days: int = typer.Option(
        30, "--days", "-d", help="Number of days to include in usage summary"
    ),
    detailed: bool = typer.Option(
        False, "--detailed", help="Show detailed usage breakdown"
    ),
) -> None:
    """Show LLM and embedding usage statistics and costs."""
    console.print(Panel("Usage Statistics", expand=False))

    # Get LLM usage from tracker
    llm_tracker = get_llm_tracker()
    if llm_tracker:
        # Get session summary from the new tracker
        session_summary = llm_tracker.get_session_summary()

        if "message" in session_summary:
            console.print(f"[{Colors.WARNING}]{session_summary['message']}[/]")
        else:
            # Create LLM usage table with semantic styling
            llm_table = Table(title="LLM Usage Summary", show_header=True)
            llm_table.add_column("Metric", style=Colors.HIGHLIGHT)
            llm_table.add_column("Value", justify="right", style=Colors.SUCCESS)

            session_stats = session_summary.get("session_summary", {})
            llm_table.add_row(
                "Total requests", str(session_stats.get("total_requests", 0))
            )
            llm_table.add_row(
                "Input tokens", f"{session_stats.get('total_input_tokens', 0):,}"
            )
            llm_table.add_row(
                "Output tokens", f"{session_stats.get('total_output_tokens', 0):,}"
            )
            llm_table.add_row(
                "Total cost", f"${session_stats.get('total_cost', 0):.4f}"
            )
            llm_table.add_row(
                "Total duration", f"{session_stats.get('total_duration_ms', 0):.0f} ms"
            )

            console.print(llm_table)

            # Show breakdown by provider if detailed
            if detailed:
                by_provider = session_summary.get("by_provider", {})

                if by_provider:
                    console.print(f"\n[{Colors.BOLD}]Breakdown by Provider:[/]")
                    provider_table = Table(show_header=True)
                    provider_table.add_column("Provider", style=Colors.HIGHLIGHT)
                    provider_table.add_column("Requests", justify="right")
                    provider_table.add_column("Input Tokens", justify="right")
                    provider_table.add_column("Output Tokens", justify="right")
                    provider_table.add_column(
                        "Cost", justify="right", style=Colors.SUCCESS
                    )

                    for provider, stats in sorted(by_provider.items()):
                        provider_table.add_row(
                            provider,
                            str(stats.get("requests", 0)),
                            f"{stats.get('input_tokens', 0):,}",
                            f"{stats.get('output_tokens', 0):,}",
                            f"${stats.get('cost', 0):.4f}",
                        )

                    console.print(provider_table)
    else:
        console.print(f"[{Colors.WARNING}]LLM usage tracking not available.[/]")

    # Show embedding usage information
    console.print(
        f"\n[{Colors.DIM}]Note: Embedding usage during ingestion is shown at the end of the ingestion process.[/]"
    )
    console.print(
        f"[{Colors.DIM}]For current session embedding costs, check the output of 'docstra ingest'.[/]"
    )


@app.command()
def explore(
    path: str = typer.Argument(".", help="Path to explore"),
    tree: bool = typer.Option(False, "--tree", help="Show as tree structure"),
    dependencies: Optional[str] = typer.Option(
        None, "--dependencies", help="Show dependencies for file"
    ),
    dependents: Optional[str] = typer.Option(
        None, "--dependents", help="Show dependents for file"
    ),
    related: Optional[str] = typer.Option(None, "--related", help="Show related files"),
    symbols: Optional[str] = typer.Option(
        None, "--symbols", help="Show symbols in file"
    ),
    structure: bool = typer.Option(False, "--structure", help="Show module structure"),
    depth: int = typer.Option(3, "--depth", help="Maximum depth to explore"),
    category: Optional[str] = typer.Option(
        None, "--category", help="Filter by module category"
    ),
    format_type: str = typer.Option(
        "table", "--format", help="Output format (table, json, tree)"
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
) -> None:
    """Explore repository structure and relationships."""
    from docstra.core.services.repository_explorer_service import (
        RepositoryExplorerService,
    )

    display_docstra_header()

    try:
        user_config = load_or_init_config(config_path)
        explorer_service = RepositoryExplorerService(user_config, console)

        if dependencies:
            # Show dependencies for a specific file
            console.print(
                f"[{Colors.INFO_BOLD}]Analyzing dependencies for:[/] {dependencies}"
            )
            relationships = explorer_service.get_file_relationships(dependencies)
            explorer_service.display_file_relationships(relationships)

        elif dependents:
            # Show dependents for a specific file
            console.print(
                f"[{Colors.INFO_BOLD}]Analyzing dependents for:[/] {dependents}"
            )
            relationships = explorer_service.get_file_relationships(dependents)

            # Create a table specifically for dependents
            if relationships["dependents"]:
                deps_table = Table(
                    title=f"Files that depend on {os.path.basename(dependents)}",
                    show_header=True,
                    header_style="bold green",
                )
                deps_table.add_column("File", style="green")
                deps_table.add_column("Full Path", style="dim")

                for dep in relationships["dependents"]:
                    deps_table.add_row(os.path.basename(dep), dep)

                console.print(deps_table)
            else:
                console.print(f"[{Colors.WARNING}]No files depend on {dependents}[/]")

        elif related:
            # Show related files
            console.print(f"[{Colors.INFO_BOLD}]Finding files related to:[/] {related}")
            relationships = explorer_service.get_file_relationships(related)

            if relationships["related_files"]:
                related_table = Table(
                    title=f"Files related to {os.path.basename(related)}",
                    show_header=True,
                    header_style="bold cyan",
                )
                related_table.add_column("File", style="cyan")
                related_table.add_column("Full Path", style="dim")

                for rel_file in relationships["related_files"]:
                    related_table.add_row(os.path.basename(rel_file), rel_file)

                console.print(related_table)
            else:
                console.print(
                    f"[{Colors.WARNING}]No related files found for {related}[/]"
                )

        elif symbols:
            # Show symbols in file
            console.print(f"[{Colors.INFO_BOLD}]Analyzing symbols in:[/] {symbols}")
            relationships = explorer_service.get_file_relationships(symbols)

            if relationships["symbols"]:
                symbols_table = Table(
                    title=f"Symbols in {os.path.basename(symbols)}",
                    show_header=True,
                    header_style="bold magenta",
                )
                symbols_table.add_column("Symbol", style="magenta")
                symbols_table.add_column("Type", style="yellow")

                # Get detailed symbol info from code index if available
                file_metadata = (
                    explorer_service.code_index.get_file_metadata(symbols)
                    if explorer_service.code_index
                    else None
                )

                if file_metadata:
                    for function in file_metadata.get("functions", []):
                        symbols_table.add_row(function, "Function")
                    for class_name in file_metadata.get("classes", []):
                        symbols_table.add_row(class_name, "Class")
                else:
                    for symbol in relationships["symbols"]:
                        symbols_table.add_row(symbol, "Symbol")

                console.print(symbols_table)
            else:
                console.print(f"[{Colors.WARNING}]No symbols found in {symbols}[/]")

        else:
            # Show general structure
            console.print(
                f"[{Colors.INFO_BOLD}]Exploring repository structure:[/] {path}"
            )

            if structure or tree:
                structure_data = explorer_service.explore_structure(path, depth, tree)

                if tree and "tree" in structure_data:
                    # Display as Rich tree
                    from rich.tree import Tree

                    repo_tree = Tree(
                        f"[{Colors.BOLD}]Repository: {os.path.basename(path)}[/]"
                    )
                    _build_rich_tree(structure_data["tree"], repo_tree)
                    console.print(repo_tree)

                elif "flat" in structure_data:
                    flat_data = structure_data["flat"]

                    # Show directory summary
                    if flat_data["directories"]:
                        dir_table = Table(
                            title="Directories",
                            show_header=True,
                            header_style="bold blue",
                        )
                        dir_table.add_column("Directory", style="blue")
                        dir_table.add_column("Files", justify="right", style="green")
                        dir_table.add_column("Depth", justify="right", style="dim")

                        for directory in flat_data["directories"][:20]:  # Show first 20
                            dir_table.add_row(
                                os.path.basename(directory["name"]) or ".",
                                str(directory["children_count"]),
                                str(directory["depth"]),
                            )

                        console.print(dir_table)

                    # Show file summary
                    if flat_data["files"]:
                        file_table = Table(
                            title="Files", show_header=True, header_style="bold green"
                        )
                        file_table.add_column("File", style="green")
                        file_table.add_column("Language", style="yellow")
                        file_table.add_column("Size", justify="right", style="cyan")
                        file_table.add_column(
                            "Symbols", justify="right", style="magenta"
                        )

                        for file_info in flat_data["files"][:20]:  # Show first 20
                            size_str = (
                                f"{file_info['size'] / 1024:.1f}KB"
                                if file_info["size"]
                                else "N/A"
                            )
                            file_table.add_row(
                                os.path.basename(file_info["name"]),
                                file_info["language"] or "Unknown",
                                size_str,
                                str(file_info["symbols"]),
                            )

                        if len(flat_data["files"]) > 20:
                            console.print(
                                f"[{Colors.DIM}]... and {len(flat_data['files']) - 20} more files[/]"
                            )

                        console.print(file_table)
            else:
                # Default overview
                structure_data = explorer_service.explore_structure(path, depth, False)
                stats = structure_data.get("statistics", {})

                overview_table = Table(
                    title="Repository Overview",
                    show_header=True,
                    header_style="bold cyan",
                )
                overview_table.add_column("Metric", style="cyan")
                overview_table.add_column("Value", justify="right", style="white")

                overview_table.add_row("Total Files", str(stats.get("total_files", 0)))
                overview_table.add_row(
                    "Total Lines", f"{stats.get('total_lines', 0):,}"
                )
                overview_table.add_row(
                    "Languages", str(len(stats.get("languages", {})))
                )
                overview_table.add_row(
                    "Modules", str(len(stats.get("module_sizes", {})))
                )

                console.print(overview_table)

                # Show language breakdown
                languages = stats.get("languages", {})
                if languages:
                    lang_table = Table(
                        title="Languages", show_header=True, header_style="bold yellow"
                    )
                    lang_table.add_column("Language", style="yellow")
                    lang_table.add_column("Files", justify="right", style="green")

                    for language, count in sorted(
                        languages.items(), key=lambda x: x[1], reverse=True
                    ):
                        lang_table.add_row(language, str(count))

                    console.print(lang_table)

    except Exception as e:
        console.print(f"[{Colors.ERROR_BOLD}]Error exploring repository: {e}[/]")
        raise typer.Exit(1)


def _build_rich_tree(node_data: Dict[str, Any], tree) -> None:
    """Build Rich tree from node data recursively.

    Args:
        node_data: Node data dictionary
        tree: Rich tree object to add to
    """
    if "children" in node_data:
        for name, child_data in node_data["children"].items():
            if child_data.get("type") == "file":
                language = child_data.get("language", "")
                symbols = child_data.get("symbols", 0)
                branch = tree.add(
                    f"[{Colors.SUCCESS}]📄 {name}[/] [{Colors.DIM}]({language}, {symbols} symbols)[/]"
                )
            else:
                branch = tree.add(f"[{Colors.HIGHLIGHT}]📁 {name}[/]")
                if "children" in child_data:
                    _build_rich_tree(child_data, branch)


@app.command()
def metrics(
    overview: bool = typer.Option(False, "--overview", help="Show repository overview"),
    complexity: bool = typer.Option(
        False, "--complexity", help="Show complexity analysis"
    ),
    by_module: bool = typer.Option(
        False, "--by-module", help="Group metrics by module"
    ),
    dependencies: bool = typer.Option(
        False, "--dependencies", help="Analyze dependencies"
    ),
    cycles: bool = typer.Option(False, "--cycles", help="Find dependency cycles"),
    coupling: bool = typer.Option(False, "--coupling", help="Analyze module coupling"),
    export: Optional[str] = typer.Option(
        None, "--export", help="Export metrics to file"
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    codebase_path: str = typer.Option(".", "--codebase", help="Path to codebase"),
) -> None:
    """Generate comprehensive code metrics and analysis."""
    from docstra.core.services.repository_explorer_service import (
        RepositoryExplorerService,
    )
    from docstra.core.services.metrics_service import MetricsService

    display_docstra_header()

    try:
        user_config = load_or_init_config(config_path)

        # Load repository components
        explorer_service = RepositoryExplorerService(user_config, console)
        explorer_service._load_components(codebase_path)

        if not explorer_service.repo_map or not explorer_service.code_index:
            console.print(
                f"[{Colors.ERROR_BOLD}]Repository not indexed. Run 'docstra ingest' first.[/]"
            )
            raise typer.Exit(1)

        metrics_service = MetricsService(
            explorer_service.repo_map, explorer_service.code_index, console
        )

        if overview:
            console.print(f"[{Colors.INFO_BOLD}]Calculating repository overview...[/]")
            metrics_data = metrics_service.calculate_repository_overview()
            metrics_service.display_repository_overview(metrics_data)

        elif cycles:
            console.print(f"[{Colors.INFO_BOLD}]Detecting dependency cycles...[/]")
            cycles_found = metrics_service.detect_dependency_cycles()
            metrics_service.display_dependency_cycles(cycles_found)

        elif complexity:
            console.print(f"[{Colors.INFO_BOLD}]Analyzing code complexity...[/]")
            metrics_data = metrics_service.calculate_repository_overview()
            complexity_analysis = metrics_data["complexity_analysis"]

            # Display complexity metrics
            complexity_table = Table(
                title="Complexity Analysis", show_header=True, header_style="bold red"
            )
            complexity_table.add_column("Metric", style="red")
            complexity_table.add_column("Value", justify="right", style="white")

            complexity_table.add_row(
                "Total Files", str(complexity_analysis.get("total_files", 0))
            )
            complexity_table.add_row(
                "Average Complexity",
                f"{complexity_analysis.get('average_complexity', 0):.2f}",
            )
            complexity_table.add_row(
                "Max Complexity", str(complexity_analysis.get("max_complexity", 0))
            )
            complexity_table.add_row(
                "High Complexity Files",
                str(complexity_analysis.get("high_complexity_files", 0)),
            )

            console.print(complexity_table)

            # Show complexity distribution
            distribution = complexity_analysis.get("complexity_distribution", {})
            if distribution:
                dist_table = Table(
                    title="Complexity Distribution",
                    show_header=True,
                    header_style="bold orange",
                )
                dist_table.add_column("Range", style="orange")
                dist_table.add_column("Files", justify="right", style="white")

                for range_name, count in distribution.items():
                    dist_table.add_row(range_name, str(count))

                console.print(dist_table)

        elif dependencies:
            console.print(f"[{Colors.INFO_BOLD}]Analyzing dependencies...[/]")
            metrics_data = metrics_service.calculate_repository_overview()
            dep_analysis = metrics_data["dependency_analysis"]

            # Display dependency metrics
            dep_table = Table(
                title="Dependency Analysis", show_header=True, header_style="bold cyan"
            )
            dep_table.add_column("Metric", style="cyan")
            dep_table.add_column("Value", justify="right", style="white")

            dep_table.add_row(
                "Files with Dependencies",
                str(dep_analysis.get("total_files_with_deps", 0)),
            )
            dep_table.add_row(
                "Total Dependencies", str(dep_analysis.get("total_dependencies", 0))
            )
            dep_table.add_row(
                "Avg Dependencies per File",
                f"{dep_analysis.get('average_dependencies_per_file', 0):.2f}",
            )
            dep_table.add_row(
                "Dependency Cycles", str(len(dep_analysis.get("dependency_cycles", [])))
            )

            console.print(dep_table)

            # Show highly coupled files
            highly_coupled = dep_analysis.get("highly_coupled_files", [])
            if highly_coupled:
                coupled_table = Table(
                    title="Highly Coupled Files",
                    show_header=True,
                    header_style="bold yellow",
                )
                coupled_table.add_column("File", style="yellow")
                coupled_table.add_column("Dependencies", justify="right", style="red")

                for file_info in highly_coupled[:10]:
                    coupled_table.add_row(
                        os.path.basename(file_info["file"]),
                        str(file_info["dependency_count"]),
                    )

                console.print(coupled_table)

        elif by_module:
            console.print(f"[{Colors.INFO_BOLD}]Analyzing metrics by module...[/]")
            metrics_data = metrics_service.calculate_repository_overview()
            module_breakdown = metrics_data["module_breakdown"]

            # Display module breakdown
            module_table = Table(
                title="Module Breakdown", show_header=True, header_style="bold green"
            )
            module_table.add_column("Module Category", style="green")
            module_table.add_column("Files", justify="right", style="cyan")
            module_table.add_column("Languages", style="yellow")
            module_table.add_column("Total Symbols", justify="right", style="magenta")
            module_table.add_column("Avg Symbols/File", justify="right", style="white")

            for category, data in module_breakdown.items():
                languages_str = ", ".join(
                    data["languages"][:3]
                )  # Show first 3 languages
                if len(data["languages"]) > 3:
                    languages_str += f" (+{len(data['languages']) - 3})"

                module_table.add_row(
                    category,
                    str(data["file_count"]),
                    languages_str,
                    str(data["total_symbols"]),
                    f"{data['avg_symbols_per_file']:.1f}",
                )

            console.print(module_table)

        else:
            # Default: show basic metrics overview
            console.print(f"[{Colors.INFO_BOLD}]Calculating basic metrics...[/]")
            metrics_data = metrics_service.calculate_repository_overview()

            # Show repository statistics
            metrics_service.display_repository_overview(metrics_data)

            # Show brief complexity info
            complexity_analysis = metrics_data["complexity_analysis"]
            if complexity_analysis.get("total_files", 0) > 0:
                console.print(f"\n[{Colors.BOLD}]Complexity Summary:[/]")
                console.print(
                    f"  • Average complexity: [{Colors.HIGHLIGHT}]{complexity_analysis.get('average_complexity', 0):.2f}[/]"
                )
                console.print(
                    f"  • High complexity files: [{Colors.WARNING}]{complexity_analysis.get('high_complexity_files', 0)}[/]"
                )

            # Show dependency cycles if any
            dep_analysis = metrics_data["dependency_analysis"]
            cycle_count = len(dep_analysis.get("dependency_cycles", []))
            if cycle_count > 0:
                console.print(
                    f"\n[{Colors.WARNING_BOLD}]⚠️  Found {cycle_count} dependency cycles![/]"
                )
                console.print(f"[{Colors.DIM}]Run with --cycles to see details[/]")
            else:
                console.print(
                    f"\n[{Colors.SUCCESS}]✅ No dependency cycles detected[/]"
                )

        # Export metrics if requested
        if export:
            import json

            metrics_data = metrics_service.calculate_repository_overview()

            try:
                with open(export, "w") as f:
                    json.dump(metrics_data, f, indent=2, default=str)
                console.print(
                    f"\n[{Colors.SUCCESS}]📄 Metrics exported to: {export}[/]"
                )
            except Exception as e:
                console.print(f"\n[{Colors.ERROR}]Failed to export metrics: {e}[/]")

    except Exception as e:
        console.print(f"[{Colors.ERROR_BOLD}]Error calculating metrics: {e}[/]")
        raise typer.Exit(1)


@app.command()
def search(
    pattern: str = typer.Argument(..., help="Pattern to search for"),
    regex: bool = typer.Option(False, "--regex", help="Use regex pattern search"),
    symbols: bool = typer.Option(False, "--symbols", help="Search in symbols only"),
    semantic: bool = typer.Option(False, "--semantic", help="Use semantic search"),
    symbol: Optional[str] = typer.Option(
        None, "--symbol", help="Search for specific symbol"
    ),
    imports: Optional[str] = typer.Option(None, "--imports", help="Find import usages"),
    similar: Optional[str] = typer.Option(
        None, "--similar", help="Find files similar to given file"
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    codebase_path: str = typer.Option(".", "--codebase", help="Path to codebase"),
    n_results: int = typer.Option(
        10, "--results", "-n", help="Number of results to show"
    ),
) -> None:
    """Specialized search command for finding code elements."""
    from docstra.core.services.repository_explorer_service import (
        RepositoryExplorerService,
    )

    display_docstra_header()

    try:
        user_config = load_or_init_config(config_path)

        # Load repository components
        explorer_service = RepositoryExplorerService(user_config, console)
        explorer_service._load_components(codebase_path)

        if not explorer_service.code_index:
            console.print(
                f"[{Colors.ERROR_BOLD}]Code index not available. Run 'docstra ingest' first.[/]"
            )
            raise typer.Exit(1)

        if symbol:
            # Search for specific symbol
            console.print(f"[{Colors.INFO_BOLD}]Searching for symbol:[/] {symbol}")
            results = explorer_service.code_index.search_symbol(symbol)

            if results:
                symbol_table = Table(
                    title=f"Symbol '{symbol}' found in:",
                    show_header=True,
                    header_style="bold magenta",
                )
                symbol_table.add_column("File", style="magenta")
                symbol_table.add_column("Line", justify="right", style="cyan")
                symbol_table.add_column("Language", style="yellow")

                for result in results[:n_results]:
                    symbol_table.add_row(
                        os.path.basename(result["filepath"]),
                        str(result.get("line", "N/A")),
                        result.get("language", "Unknown"),
                    )

                console.print(symbol_table)

                if len(results) > n_results:
                    console.print(
                        f"[{Colors.DIM}]... and {len(results) - n_results} more results[/]"
                    )
            else:
                console.print(f"[{Colors.WARNING}]Symbol '{symbol}' not found[/]")

        elif imports:
            # Search for import usages
            console.print(f"[{Colors.INFO_BOLD}]Searching for import:[/] {imports}")
            import_results = explorer_service.code_index.search_files_by_import(imports)

            if import_results:
                import_table = Table(
                    title=f"Files importing '{imports}':",
                    show_header=True,
                    header_style="bold cyan",
                )
                import_table.add_column("File", style="cyan")
                import_table.add_column("Full Path", style="dim")

                for result_file in import_results[:n_results]:
                    import_table.add_row(os.path.basename(result_file), result_file)

                console.print(import_table)

                if len(import_results) > n_results:
                    console.print(
                        f"[{Colors.DIM}]... and {len(import_results) - n_results} more results[/]"
                    )
            else:
                console.print(f"[{Colors.WARNING}]Import '{imports}' not found[/]")

        elif similar:
            # Find similar files
            console.print(f"[{Colors.INFO_BOLD}]Finding files similar to:[/] {similar}")
            related_files = explorer_service.code_index.get_related_files(similar)

            if related_files:
                similar_table = Table(
                    title=f"Files similar to '{os.path.basename(similar)}':",
                    show_header=True,
                    header_style="bold green",
                )
                similar_table.add_column("File", style="green")
                similar_table.add_column("Full Path", style="dim")

                for similar_file in related_files[:n_results]:
                    similar_table.add_row(os.path.basename(similar_file), similar_file)

                console.print(similar_table)

                if len(related_files) > n_results:
                    console.print(
                        f"[{Colors.DIM}]... and {len(related_files) - n_results} more results[/]"
                    )
            else:
                console.print(
                    f"[{Colors.WARNING}]No files similar to '{similar}' found[/]"
                )

        else:
            # Full-text search
            console.print(f"[{Colors.INFO_BOLD}]Searching for:[/] {pattern}")
            results = explorer_service.code_index.full_text_search(pattern)

            if results:
                search_table = Table(
                    title=f"Search results for '{pattern}':",
                    show_header=True,
                    header_style="bold white",
                )
                search_table.add_column("File", style="white")
                search_table.add_column("Language", style="yellow")
                search_table.add_column("Matches", justify="right", style="green")
                search_table.add_column("Sample Match", style="dim")

                for result in results[:n_results]:
                    matches = result.get("matches", [])
                    sample_match = (
                        matches[0]["line_content"][:50] + "..." if matches else ""
                    )

                    search_table.add_row(
                        os.path.basename(result["filepath"]),
                        result.get("language", "Unknown"),
                        str(len(matches)),
                        sample_match,
                    )

                console.print(search_table)

                if len(results) > n_results:
                    console.print(
                        f"[{Colors.DIM}]... and {len(results) - n_results} more results[/]"
                    )
            else:
                console.print(f"[{Colors.WARNING}]Pattern '{pattern}' not found[/]")

    except Exception as e:
        console.print(f"[{Colors.ERROR_BOLD}]Error during search: {e}[/]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
