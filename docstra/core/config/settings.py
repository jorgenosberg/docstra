from enum import Enum
from typing import Any, Optional, List
import os
import yaml
from pathlib import Path


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
        self.load()  # Load existing configuration if available

    def update(self, **kwargs: Any) -> None:
        """Update configuration with provided key-value pairs."""
        def _update_nested(obj: Any, path_parts: List[str], value: Any) -> None:
            if len(path_parts) == 1:
                if hasattr(obj, path_parts[0]):
                    setattr(obj, path_parts[0], value)
            else:
                if hasattr(obj, path_parts[0]):
                    next_obj = getattr(obj, path_parts[0])
                    _update_nested(next_obj, path_parts[1:], value)
        
        def _update_from_dict(obj: Any, data: dict, parent_key: str = "") -> None:
            """Recursively update object from nested dictionary"""
            for key, value in data.items():
                if isinstance(value, dict):
                    # Get the nested object
                    if hasattr(obj, key):
                        nested_obj = getattr(obj, key)
                        _update_from_dict(nested_obj, value, f"{parent_key}.{key}" if parent_key else key)
                else:
                    # Set the value
                    if hasattr(obj, key):
                        setattr(obj, key, value)
        
        # Process all kwargs
        _update_from_dict(self.config, kwargs)

    def reset_to_default(self) -> None:
        """Reset configuration to default values."""
        self.config = UserConfig()

    def load(self) -> None:
        """Load configuration from file."""
        if os.path.exists(self.config_path):
            self.config.load_from_file(self.config_path)
    
    def save(self) -> None:
        """Save configuration to file."""
        self.config.save_to_file(self.config_path)


class UserConfig:
    def __init__(self) -> None:
        self.model = ModelConfig()
        self.embedding = EmbeddingConfig()
        self.storage = StorageConfig()
        self.processing = ProcessingConfig()
        self.ingestion = IngestionConfig()
        self.documentation = DocumentationConfig()

    def save_to_file(self, path: str) -> None:
        """Save configuration to YAML file."""
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert configuration to dictionary
        config_dict = {
            'model': {
                'provider': str(self.model.provider),
                'model_name': self.model.model_name,
                'model_name_chat': self.model.model_name_chat,
                'api_key': self.model.api_key,
                'api_base': self.model.api_base,
                'max_tokens': self.model.max_tokens,
                'temperature': self.model.temperature,
                'model_path': self.model.model_path,
                'device': self.model.device,
                'context_window': self.model.context_window,
                'context_mode': self.model.context_mode,
            },
            'embedding': {
                'provider': self.embedding.provider,
                'model_name': self.embedding.model_name,
                'api_key': self.embedding.api_key,
            },
            'storage': {
                'persist_directory': self.storage.persist_directory,
            },
            'processing': {
                'chunk_size': self.processing.chunk_size,
                'chunk_overlap': self.processing.chunk_overlap,
                'exclude_patterns': self.processing.exclude_patterns,
            },
            'ingestion': {
                'include_dirs': self.ingestion.include_dirs,
                'exclude_patterns': self.ingestion.exclude_patterns,
            },
            'documentation': self.documentation.model_dump(),
        }
        
        # Write to YAML file
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def load_from_file(self, path: str) -> None:
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if not config_dict:
            return
        
        # Update model configuration
        if 'model' in config_dict:
            model_data = config_dict['model']
            if 'provider' in model_data:
                self.model.provider = ModelProvider(model_data['provider'])
            if 'model_name' in model_data:
                self.model.model_name = model_data['model_name']
            if 'model_name_chat' in model_data:
                self.model.model_name_chat = model_data['model_name_chat']
            if 'api_key' in model_data:
                self.model.api_key = model_data['api_key']
            if 'api_base' in model_data:
                self.model.api_base = model_data['api_base']
            if 'max_tokens' in model_data:
                self.model.max_tokens = model_data['max_tokens']
            if 'temperature' in model_data:
                self.model.temperature = model_data['temperature']
            if 'model_path' in model_data:
                self.model.model_path = model_data['model_path']
            if 'device' in model_data:
                self.model.device = model_data['device']
            if 'context_window' in model_data:
                self.model.context_window = model_data['context_window']
            if 'context_mode' in model_data:
                self.model.context_mode = model_data['context_mode']
        
        # Update embedding configuration
        if 'embedding' in config_dict:
            embedding_data = config_dict['embedding']
            if 'provider' in embedding_data:
                self.embedding.provider = embedding_data['provider']
            if 'model_name' in embedding_data:
                self.embedding.model_name = embedding_data['model_name']
            if 'api_key' in embedding_data:
                self.embedding.api_key = embedding_data['api_key']
        
        # Update other configurations as needed
        if 'storage' in config_dict:
            storage_data = config_dict['storage']
            if 'persist_directory' in storage_data:
                self.storage.persist_directory = storage_data['persist_directory']
        
        if 'processing' in config_dict:
            processing_data = config_dict['processing']
            if 'chunk_size' in processing_data:
                self.processing.chunk_size = processing_data['chunk_size']
            if 'chunk_overlap' in processing_data:
                self.processing.chunk_overlap = processing_data['chunk_overlap']
            if 'exclude_patterns' in processing_data:
                self.processing.exclude_patterns = processing_data['exclude_patterns']
