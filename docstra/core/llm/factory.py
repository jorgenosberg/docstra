"""Factory for constructing LLM clients from user configuration."""

from __future__ import annotations

from typing import Any, Optional

from docstra.core.config.settings import ModelProvider, UserConfig


def create_llm_client(config: UserConfig, model_name: Optional[str] = None) -> Any:
    """Create an LLM client for the configured provider.

    The concrete clients share the LLMClient interface by convention
    (duck-typed, not subclassed), so the return type stays loose.

    Args:
        config: User configuration holding provider and model settings
        model_name: Optional model override (for task-tier routing); defaults
            to the configured model_name

    Returns:
        An LLM client for the configured provider
    """
    provider = config.model.provider
    effective_model = model_name or config.model.model_name

    if provider == ModelProvider.ANTHROPIC:
        from docstra.core.llm.anthropic import AnthropicClient

        return AnthropicClient(
            model_name=effective_model,
            api_key=config.model.api_key,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
        )
    if provider == ModelProvider.OPENAI:
        from docstra.core.llm.openai import OpenAIClient

        return OpenAIClient(
            model_name=effective_model,
            api_key=config.model.api_key,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
        )
    if provider == ModelProvider.OLLAMA:
        from docstra.core.llm.ollama import OllamaClient

        return OllamaClient(
            model_name=effective_model,
            api_base=config.model.api_base or "http://localhost:11434",
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
            validate_connection=False,
        )
    if provider == ModelProvider.LOCAL:
        from docstra.core.llm.local import LocalModelClient

        return LocalModelClient(
            model_name=effective_model,
            model_path=config.model.model_path,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
            device=config.model.device,
        )
    raise ValueError(f"Unsupported model provider: {provider}")
