"""Tests for local-first defaults, the LLM client factory, and model routing."""

from __future__ import annotations

from pathlib import Path

from docstra.core.config.settings import (
    DEFAULT_OLLAMA_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
    ModelProvider,
    UserConfig,
)
from docstra.core.document_processing.document import (
    Document,
    DocumentMetadata,
    DocumentType,
)
from docstra.core.documentation.generator import DocumentationGenerator
from docstra.core.llm.factory import create_llm_client
from docstra.core.llm.ollama import OllamaClient, strip_think_blocks


def test_fresh_config_is_local_first() -> None:
    config = UserConfig()

    assert config.model.provider == ModelProvider.OLLAMA
    assert config.model.model_name == DEFAULT_OLLAMA_MODEL
    assert config.model.model_name_overview is None
    assert config.embedding.provider == "ollama"
    assert config.embedding.model_name == DEFAULT_OLLAMA_EMBEDDING_MODEL


def test_config_round_trip_preserves_new_fields(tmp_path: Path) -> None:
    config = UserConfig()
    config.model.model_name_overview = "qwen3:32b"
    config.documentation.output_dir = "./site"
    config.documentation.max_workers_default = 3
    config.ingestion.exclude_patterns = ["vendor/**"]

    path = tmp_path / "config.yaml"
    config.save_to_file(str(path))

    loaded = UserConfig()
    loaded.load_from_file(str(path))

    assert loaded.model.model_name_overview == "qwen3:32b"
    assert loaded.documentation.output_dir == "./site"
    assert loaded.documentation.max_workers_default == 3
    assert loaded.ingestion.exclude_patterns == ["vendor/**"]


def test_strip_think_blocks() -> None:
    raw = "<think>\nreasoning about the code\n</think>\n# Real documentation\n\nBody."
    assert strip_think_blocks(raw) == "# Real documentation\n\nBody."

    assert strip_think_blocks("no reasoning here") == "no reasoning here"
    assert strip_think_blocks("<think>a</think>x<think>b</think>y") == "xy"


def test_factory_builds_ollama_client_with_model_override() -> None:
    config = UserConfig()

    default_client = create_llm_client(config)
    assert isinstance(default_client, OllamaClient)
    assert default_client.model_name == DEFAULT_OLLAMA_MODEL

    overview_client = create_llm_client(config, model_name="qwen3:32b")
    assert isinstance(overview_client, OllamaClient)
    assert overview_client.model_name == "qwen3:32b"


class RecordingClient:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def document_code(self, code: str, language: str, additional_context: str) -> str:
        del code, language, additional_context
        self.calls += 1
        return f"# Written by {self.name}"


def _make_document(repo: Path, name: str) -> Document:
    content = "def f() -> None:\n    pass\n"
    filepath = repo / name
    filepath.write_text(content)
    return Document(
        content=content,
        metadata=DocumentMetadata(
            filepath=str(filepath),
            language=DocumentType.PYTHON,
            size_bytes=len(content),
            last_modified=1.0,
            line_count=2,
            imports=[],
            classes=[],
            functions=["f"],
            symbols={},
        ),
        chunks=[],
    )


def test_overview_pages_use_overview_client_and_file_pages_do_not(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    document = _make_document(repo, "app.py")

    file_client = RecordingClient("file-model")
    overview_client = RecordingClient("overview-model")
    generator = DocumentationGenerator(
        llm_client=file_client,
        overview_llm_client=overview_client,
        output_dir=tmp_path / "docs_out",
    )

    assert generator.generate_documentation(
        [document], generate_guides=False, generate_api_docs=False
    )

    file_page = (tmp_path / "docs_out" / "docs" / "api" / "app.py.md").read_text(
        encoding="utf-8"
    )
    assert "Written by file-model" in file_page

    overview_page = (tmp_path / "docs_out" / "docs" / "index.md").read_text(
        encoding="utf-8"
    )
    assert "Written by overview-model" in overview_page

    module_pages = list((tmp_path / "docs_out" / "docs" / "modules").rglob("index.md"))
    assert module_pages
    assert "Written by overview-model" in module_pages[0].read_text(encoding="utf-8")

    assert file_client.calls == 1  # one file page
    assert overview_client.calls == 2  # overview + one module page


def test_generator_without_overview_client_uses_default_for_everything(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    document = _make_document(repo, "app.py")

    client = RecordingClient("only-model")
    generator = DocumentationGenerator(
        llm_client=client, output_dir=tmp_path / "docs_out"
    )

    assert generator.generate_documentation(
        [document], generate_guides=False, generate_api_docs=False
    )
    assert client.calls == 3  # overview + module + file page
