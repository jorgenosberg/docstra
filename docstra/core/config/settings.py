from enum import Enum
from typing import Any, Optional, List


class ModelProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    LOCAL = "local"
    
    def __str__(self) -> str:
        """Return the enum value instead of the full enum representation."""
        return self.value


class ModelConfig:
    def __init__(
        self,
        provider: ModelProvider = ModelProvider.OLLAMA,
        model_name: str = "llama3.2",
        model_name_chat: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        model_path: Optional[str] = None,
        device: str = "auto",
        context_window: Optional[int] = None,
        context_mode: str = "balanced",
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.model_name_chat = model_name_chat or model_name
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_path = model_path
        self.device = device
        self.context_window = context_window
        self.context_mode = context_mode


class EmbeddingConfig:
    def __init__(
        self,
        provider: str = "huggingface",
        model_name: str = "all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key


class StorageConfig:
    def __init__(self, persist_directory: str = ".docstra") -> None:
        self.persist_directory = persist_directory


class DocumentationConfig:
    def __init__(
        self,
        include_dirs: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        output_dir: str = "./docs",
        format: str = "markdown",
        theme: str = "default",
        project_name: Optional[str] = None,
        project_description: Optional[str] = None,
        project_version: str = "0.1.0",
        documentation_structure: str = "file_based",
        module_doc_depth: str = "full",
        llm_style_prompt: Optional[str] = None,
        max_workers_ollama: int = 1,
        max_workers_api: int = 4,
        max_workers_default: Optional[int] = None,
    ) -> None:
        self.include_dirs = include_dirs
        self.exclude_patterns = exclude_patterns
        self.output_dir = output_dir
        self.format = format
        self.theme = theme
        self.project_name = project_name
        self.project_description = project_description
        self.project_version = project_version
        self.documentation_structure = documentation_structure
        self.module_doc_depth = module_doc_depth
        self.llm_style_prompt = llm_style_prompt
        self.max_workers_ollama = max_workers_ollama
        self.max_workers_api = max_workers_api
        self.max_workers_default = max_workers_default

    def model_dump(self) -> dict:
        return self.__dict__


class IngestionConfig:
    def __init__(
        self,
        include_dirs: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> None:
        self.include_dirs = include_dirs
        self.exclude_patterns = exclude_patterns


class ProcessingConfig:
    def __init__(
        self,
        chunk_size: int = 100,
        chunk_overlap: int = 20,
        exclude_patterns: Optional[List[str]] = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.exclude_patterns = exclude_patterns or []


class ConfigManager:
    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = config_path or "./.docstra/config.yaml"
        self.config = UserConfig()

    def update(self, **kwargs: Any) -> None:
        pass

    def reset_to_default(self) -> None:
        pass

    def save(self) -> None:
        pass


class UserConfig:
    def __init__(self) -> None:
        self.model = ModelConfig()
        self.embedding = EmbeddingConfig()
        self.storage = StorageConfig()
        self.processing = ProcessingConfig()
        self.ingestion = IngestionConfig()
        self.documentation = DocumentationConfig()

    def save_to_file(self, path: str) -> None:
        pass
