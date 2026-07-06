# File: ./docstra/core/config/wizard.py

import os
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from docstra.core.config.settings import (
    DEFAULT_OLLAMA_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
    ConfigManager,
    ModelProvider,
    UserConfig,
)
from docstra.core.utils.colors import Colors


class ConfigScope(str, Enum):
    """Scope of configuration settings."""

    GLOBAL = "global"
    LOCAL = "local"
    BOTH = "both"


class ConfigField:
    """Representation of a configuration field for the wizard."""

    def __init__(
        self,
        name: str,
        path: str,
        description: str,
        field_type: type = str,
        choices: Optional[List[str]] = None,
        default: Any = None,
        scope: ConfigScope = ConfigScope.BOTH,
        required: bool = False,
        sensitive: bool = False,
        advanced: bool = False,
        validator: Optional[Callable[[Any], Tuple[bool, str]]] = None,
    ):
        """Initialize a configuration field.

        Args:
            name: Display name of the field
            path: Dot-separated path in the config (e.g., "model.api_key")
            description: Description of the field
            field_type: Type of the field (str, int, float, bool)
            choices: List of valid choices for the field
            default: Default value for the field
            scope: Whether this field applies to global, local, or both configs
            required: Whether this field is required
            sensitive: Whether this field contains sensitive data (like API keys)
            advanced: Whether this is an advanced setting
            validator: Optional function to validate the value
        """
        self.name = name
        self.path = path
        self.description = description
        self.field_type = field_type
        self.choices = choices
        self.default = default
        self.scope = scope
        self.required = required
        self.sensitive = sensitive
        self.advanced = advanced
        self.validator = validator

    def get_value_from_config(self, config: UserConfig) -> Any:
        """Get value from config using the path.

        Args:
            config: Configuration object

        Returns:
            Field value from the config
        """
        parts = self.path.split(".")
        obj = config

        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None

        if hasattr(obj, parts[-1]):
            return getattr(obj, parts[-1])

        return None


class ConfigWizard:
    """Interactive wizard for configuration setup."""

    def __init__(
        self,
        console: Console,
        config_path: Optional[str] = None,
        local_path: Optional[str] = None,
    ):
        """Initialize the configuration wizard.

        Args:
            console: Rich console for UI
            config_path: Optional path to the global configuration file
            local_path: Optional path to local project directory
        """
        self.console = console
        self.global_config_manager = ConfigManager(config_path)

        # Create local config manager if a local path is provided
        self.local_config_manager = None
        if local_path:
            local_config_path = os.path.join(local_path, ".docstra", "config.yaml")
            # Create local config directory if it doesn't exist
            os.makedirs(os.path.dirname(local_config_path), exist_ok=True)

            # Try to load local config, or create a new one
            try:
                self.local_config_manager = ConfigManager(local_config_path)
            except Exception:
                # No local config exists yet, we'll create one during the wizard
                self.local_config_manager = ConfigManager(local_config_path)

        # Define configuration fields
        self.fields = self._define_fields()

    def _define_fields(self) -> List[ConfigField]:
        """Define all configuration fields.

        Returns:
            List of configuration fields
        """
        fields = []

        # Model configuration
        fields.append(
            ConfigField(
                name="Model Provider",
                path="model.provider",
                description="LLM provider to use (anthropic, openai, ollama, local)",
                choices=[p.value for p in ModelProvider],
                default=ModelProvider.OLLAMA.value,
                required=True,
                scope=ConfigScope.BOTH,
                validator=validate_model_provider,
            )
        )

        fields.append(
            ConfigField(
                name="Model Name",
                path="model.model_name",
                description="Name of the model to use",
                default=DEFAULT_OLLAMA_MODEL,
                required=True,
                scope=ConfigScope.BOTH,
            )
        )

        fields.append(
            ConfigField(
                name="Model API Key",
                path="model.api_key",
                description="API key for the model provider (if needed)",
                sensitive=True,
                scope=ConfigScope.GLOBAL,
            )
        )

        fields.append(
            ConfigField(
                name="Model API Base",
                path="model.api_base",
                description="Base URL for the model API (if needed)",
                default=(
                    "http://localhost:11434"
                    if ModelProvider.OLLAMA.value in str(fields[0].default)
                    else None
                ),
                scope=ConfigScope.BOTH,
                advanced=True,
            )
        )

        fields.append(
            ConfigField(
                name="Maximum Tokens",
                path="model.max_tokens",
                description="Maximum number of tokens to generate",
                field_type=int,
                default=4000,
                scope=ConfigScope.BOTH,
                advanced=True,
            )
        )

        fields.append(
            ConfigField(
                name="Temperature",
                path="model.temperature",
                description="Temperature for generation (0.0 to 1.0)",
                field_type=float,
                default=0.7,
                scope=ConfigScope.BOTH,
                advanced=True,
            )
        )

        # Embedding configuration
        fields.append(
            ConfigField(
                name="Embedding Provider",
                path="embedding.provider",
                description="Provider for embeddings",
                choices=["ollama", "huggingface", "openai"],
                default="ollama",
                scope=ConfigScope.BOTH,
            )
        )

        fields.append(
            ConfigField(
                name="Embedding Model",
                path="embedding.model_name",
                description="Name of the embedding model (changing it requires re-running 'docstra ingest')",
                default=DEFAULT_OLLAMA_EMBEDDING_MODEL,
                scope=ConfigScope.BOTH,
            )
        )

        fields.append(
            ConfigField(
                name="Embedding API Key",
                path="embedding.api_key",
                description="API key for the embedding provider (if needed)",
                sensitive=True,
                scope=ConfigScope.GLOBAL,
            )
        )

        # Storage configuration
        fields.append(
            ConfigField(
                name="Storage Directory",
                path="storage.persist_directory",
                description="Directory to persist data",
                default=".docstra",
                scope=ConfigScope.LOCAL,
            )
        )

        # Processing configuration
        fields.append(
            ConfigField(
                name="Chunk Size",
                path="processing.chunk_size",
                description="Size of chunks in lines",
                field_type=int,
                default=100,
                scope=ConfigScope.BOTH,
                advanced=True,
            )
        )

        fields.append(
            ConfigField(
                name="Chunk Overlap",
                path="processing.chunk_overlap",
                description="Overlap between chunks in lines",
                field_type=int,
                default=20,
                scope=ConfigScope.BOTH,
                advanced=True,
            )
        )

        fields.append(
            ConfigField(
                name="Exclude Patterns",
                path="processing.exclude_patterns",
                description="Patterns to exclude from processing (comma-separated)",
                default=".git,__pycache__,.mypy_cache,.ruff_cache,.pytest_cache,node_modules,venv,.venv,env,.env,.vscode,.idea,build,dist",
                scope=ConfigScope.LOCAL,
            )
        )

        return fields

    def _get_field_default(self, field: ConfigField, scope: ConfigScope) -> Any:
        """Get the default value for a field based on existing configs.

        Args:
            field: Configuration field
            scope: Configuration scope

        Returns:
            Default value for the field
        """
        # Check local config first if it exists and is in scope
        if scope in [ConfigScope.LOCAL, ConfigScope.BOTH] and self.local_config_manager:
            local_value = field.get_value_from_config(self.local_config_manager.config)
            if local_value is not None:
                return local_value

        # Then check global config if in scope
        if scope in [ConfigScope.GLOBAL, ConfigScope.BOTH]:
            global_value = field.get_value_from_config(
                self.global_config_manager.config
            )
            if global_value is not None:
                return global_value

        # Fall back to field default
        return field.default

    def _prompt_for_field(self, field: ConfigField, scope: ConfigScope) -> Any:
        """Prompt the user for a field value.

        Args:
            field: Configuration field
            scope: Configuration scope

        Returns:
            User input for the field
        """
        # Get current value as default
        default = self._get_field_default(field, scope)

        # Format default for display
        display_default = default
        if isinstance(default, list):
            display_default = ",".join(str(x) for x in default)

        # Show field description with semantic colors
        self.console.print(f"\n[{Colors.HIGHLIGHT}]{field.name}[/]")
        self.console.print(f"[{Colors.DIM}]{field.description}[/]")

        # For sensitive fields, don't show the actual value
        if field.sensitive and default:
            display_default = "********"

        # Handle different field types
        if field.choices:
            # For fields with choices, use a menu
            value = Prompt.ask(
                "Select one",
                choices=field.choices,
                default=(
                    str(display_default)
                    if display_default in field.choices
                    else field.choices[0]
                ),
            )
        elif isinstance(field.field_type, type) and field.field_type is bool:
            # For boolean fields, use a confirmation
            prompt_default = (
                str(bool(default)).lower() if default is not None else "false"
            )
            value_str = Prompt.ask(
                "Enable? (true/false)",
                choices=["true", "false"],
                default=prompt_default,
            )
            value = value_str.lower()  # keep as str for type consistency
        elif field.field_type in [int, float]:
            # For numeric fields, parse the input
            prompt_default = str(display_default) if display_default is not None else ""
            prompt_value = Prompt.ask(
                "Enter value",
                default=prompt_default,
            )
            try:
                value = str(field.field_type(prompt_value))
            except (ValueError, TypeError):
                self.console.print(
                    f"[{Colors.ERROR}]Invalid numeric value, using default.[/]"
                )
                value = str(default)
        else:
            # For string fields and others
            prompt_default = str(display_default) if display_default is not None else ""
            if isinstance(prompt_default, bool):
                prompt_default = str(prompt_default)
            value = Prompt.ask(
                "Enter value",
                default=prompt_default,
            )

        # Validate if needed
        if field.validator and value is not None:
            result = field.validator(value)
            if isinstance(result, tuple):
                valid, message = result
            else:
                valid, message = bool(result), ""

            if not valid:
                self.console.print(
                    f"[{Colors.ERROR_BOLD}]Validation failed: {message}[/]"
                )

                # For model provider, offer alternatives
                if field.path == "model.provider":
                    self.console.print(
                        f"\n[{Colors.WARNING}]Available alternatives:[/]"
                    )
                    self.console.print(
                        f"  - [{Colors.HIGHLIGHT}]anthropic[/] (requires API key)"
                    )
                    self.console.print(
                        f"  - [{Colors.HIGHLIGHT}]openai[/] (requires API key)"
                    )
                    self.console.print(
                        f"  - [{Colors.HIGHLIGHT}]local[/] (requires local model)"
                    )

                    retry = Confirm.ask(
                        "Would you like to choose a different provider?", default=True
                    )
                    if retry:
                        return self._prompt_for_field(field, scope)

                self.console.print(
                    f"[{Colors.WARNING}]Using default value: {default}[/]"
                )
                value = str(default)
            else:
                # Show validation message (could be success or warning)
                if "Warning:" in message:
                    self.console.print(f"[{Colors.WARNING}]{message}[/]")
                else:
                    self.console.print(f"[{Colors.SUCCESS}]{message}[/]")

        # Always return a string for value
        if isinstance(value, list):
            value = ", ".join(map(str, value))
        elif not isinstance(value, str):
            value = str(value)

        return value

    def _update_config_with_value(
        self, config_manager: ConfigManager, field: ConfigField, value: Any
    ) -> None:
        """Update a config manager with a field value.

        Args:
            config_manager: Configuration manager to update
            field: Configuration field
            value: Value to set
        """
        parts = field.path.split(".")

        # Convert exclude_patterns to list if needed
        if field.path == "processing.exclude_patterns" and isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]

        # Convert boolean string to bool if needed
        if field.field_type is bool and isinstance(value, str):
            value = value.lower() == "true"

        # Convert to int/float if needed
        if field.field_type in [int, float] and isinstance(value, str):
            try:
                value = field.field_type(value)
            except (ValueError, TypeError):
                pass

        # Create nested update dictionary
        update_dict: Dict[str, Any] = {}
        current = update_dict

        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                current[part] = value
            else:
                current[part] = {}
                current = current[part]

        # Update the config
        config_manager.update(**update_dict)

    def _select_fields_to_configure(
        self, all_fields: List[ConfigField], scope: ConfigScope, advanced: bool = False
    ) -> List[ConfigField]:
        """Let the user select which fields to configure.

        Args:
            all_fields: List of all available fields
            scope: Configuration scope
            advanced: Whether to include advanced fields

        Returns:
            List of fields to configure
        """
        # Filter fields by scope and advanced status
        available_fields = [
            f
            for f in all_fields
            if f.scope in [scope, ConfigScope.BOTH] and f.advanced <= advanced
        ]

        # Create a table of available fields with semantic styling
        table = Table(title=f"Available Configuration Fields ({scope.value})")
        table.add_column("#", justify="right", style=Colors.HIGHLIGHT)
        table.add_column("Field", style=Colors.SUCCESS)
        table.add_column("Description")
        table.add_column("Current Value", style=Colors.WARNING)

        for i, field in enumerate(available_fields):
            current_value = self._get_field_default(field, scope)

            # Format display value
            if current_value is None:
                display_value = f"[{Colors.DIM}]Not set[/]"
            elif field.sensitive and current_value:
                display_value = "********"
            elif isinstance(current_value, list):
                display_value = ",".join(current_value)
            else:
                display_value = str(current_value)

            table.add_row(str(i + 1), field.name, field.description, display_value)

        self.console.print(table)

        # Let user select fields
        selection = Prompt.ask(
            "\nWhich fields would you like to configure? (comma-separated numbers, 'all', or empty to skip)",
            default="",
        )

        if not selection:
            return []
        elif selection.lower() == "all":
            return available_fields
        else:
            try:
                indices = [int(idx.strip()) - 1 for idx in selection.split(",")]
                return [
                    available_fields[idx]
                    for idx in indices
                    if 0 <= idx < len(available_fields)
                ]
            except (ValueError, IndexError):
                self.console.print(
                    f"[{Colors.ERROR}]Invalid selection, please try again.[/]"
                )
                return self._select_fields_to_configure(all_fields, scope, advanced)

    def run_config_wizard(
        self,
        scope: ConfigScope = ConfigScope.BOTH,
        fields_to_configure: Optional[List[ConfigField]] = None,
        include_advanced: bool = False,
    ) -> None:
        """Run the configuration wizard.

        Args:
            scope: Configuration scope to modify
            fields_to_configure: Optional list of fields to configure
            include_advanced: Whether to include advanced fields
        """
        self.console.print(Panel(f"Configuration Wizard ({scope.value})", expand=False))

        # Local config only works if we have a local config manager
        if (
            scope in [ConfigScope.LOCAL, ConfigScope.BOTH]
            and not self.local_config_manager
        ):
            self.console.print(
                f"[{Colors.WARNING}]No local project specified, can only configure global settings.[/]"
            )
            scope = ConfigScope.GLOBAL

        # If no fields specified, let user select them
        if not fields_to_configure:
            fields_to_configure = self._select_fields_to_configure(
                self.fields, scope, include_advanced
            )

        if not fields_to_configure:
            self.console.print(
                f"[{Colors.WARNING}]No fields selected for configuration.[/]"
            )
            return

        # Configure global settings
        if scope in [ConfigScope.GLOBAL, ConfigScope.BOTH]:
            self.console.print(f"\n[{Colors.BOLD}]Global Configuration[/]")
            global_fields = [
                f
                for f in fields_to_configure
                if f.scope in [ConfigScope.GLOBAL, ConfigScope.BOTH]
            ]

            for field in global_fields:
                value = self._prompt_for_field(field, ConfigScope.GLOBAL)
                if value is not None:
                    self._update_config_with_value(
                        self.global_config_manager, field, value
                    )

            # Save global configuration
            self.global_config_manager.save()
            self.console.print(f"[{Colors.SUCCESS}]Global configuration updated.[/]")

        # Configure local settings
        if scope in [ConfigScope.LOCAL, ConfigScope.BOTH] and self.local_config_manager:
            self.console.print(f"\n[{Colors.BOLD}]Local Project Configuration[/]")
            local_fields = [
                f
                for f in fields_to_configure
                if f.scope in [ConfigScope.LOCAL, ConfigScope.BOTH]
            ]

            for field in local_fields:
                value = self._prompt_for_field(field, ConfigScope.LOCAL)
                if value is not None:
                    self._update_config_with_value(
                        self.local_config_manager, field, value
                    )

            # Save local configuration
            self.local_config_manager.save()
            self.console.print(f"[{Colors.SUCCESS}]Local configuration updated.[/]")

    def run_init_wizard(self) -> None:
        """Run the initialization wizard for a new project."""
        self.console.print(Panel("Project Initialization Wizard", expand=False))

        if not self.local_config_manager:
            self.console.print(
                f"[{Colors.ERROR}]No local project specified, cannot initialize.[/]"
            )
            return

        # Explain the wizard
        self.console.print(
            "This wizard will guide you through setting up a new docstra project.\n"
            "It will create a local configuration based on your global settings and let you customize it."
        )

        # First check if global config exists and is valid
        has_valid_global = all(
            [
                self.global_config_manager.config.model.provider,
                self.global_config_manager.config.model.model_name,
            ]
        )

        if not has_valid_global:
            self.console.print(
                f"[{Colors.WARNING}]Global configuration is incomplete. Let's set it up first.[/]"
            )
            # Configure essential global settings
            essential_global_fields = [
                f
                for f in self.fields
                if f.required and f.scope in [ConfigScope.GLOBAL, ConfigScope.BOTH]
            ]
            self.run_config_wizard(ConfigScope.GLOBAL, essential_global_fields)

        # Now configure local settings
        self.console.print(f"\n[{Colors.BOLD}]Local Project Configuration[/]")
        self.console.print("Let's configure your local project settings.")

        # Configure all local fields
        local_fields = [
            f for f in self.fields if f.scope in [ConfigScope.LOCAL, ConfigScope.BOTH]
        ]

        for field in local_fields:
            # Only prompt if the field is required or the user wants to configure it
            if field.required or Confirm.ask(f"Configure {field.name}?", default=False):
                value = self._prompt_for_field(field, ConfigScope.LOCAL)
                if value is not None:
                    self._update_config_with_value(
                        self.local_config_manager, field, value
                    )

        # Save the local configuration
        self.local_config_manager.save()

        self.console.print(
            f"[{Colors.SUCCESS_BOLD}]Project initialized successfully![/]"
        )
        self.console.print(
            f"Local configuration saved to: [{Colors.HIGHLIGHT}]{self.local_config_manager.config_path}[/]"
        )

        # Suggest next steps
        self.console.print(f"\n[{Colors.BOLD}]Next steps:[/]")
        self.console.print("- Use 'docstra ingest .' to index your codebase")
        self.console.print("- Use 'docstra generate .' to generate documentation")
        self.console.print(
            "- Use 'docstra query \"How does this work?\"' to query your codebase"
        )


def run_config_wizard(
    console: Console,
    config_path: Optional[str] = None,
    local_path: Optional[str] = None,
    scope: str = "both",
    include_advanced: bool = False,
) -> None:
    """Run the configuration wizard.

    Args:
        console: Rich console
        config_path: Optional path to global config
        local_path: Optional path to local project
        scope: Scope of configuration to modify ("global", "local", or "both")
        include_advanced: Whether to include advanced fields
    """
    try:
        config_scope = ConfigScope(scope.lower())
    except ValueError:
        console.print(f"[{Colors.ERROR}]Invalid scope: {scope}. Using 'both'.[/]")
        config_scope = ConfigScope.BOTH

    wizard = ConfigWizard(console, config_path, local_path)
    wizard.run_config_wizard(config_scope, include_advanced=include_advanced)


def run_init_wizard(
    console: Console, local_path: str, config_path: Optional[str] = None
) -> None:
    """Run the initialization wizard for a new project.

    Args:
        console: Rich console
        local_path: Path to local project
        config_path: Optional path to global config
    """
    wizard = ConfigWizard(console, config_path, local_path)
    wizard.run_init_wizard()


def validate_model_provider(provider: str) -> Tuple[bool, str]:
    """Validate that the selected model provider is available.

    Args:
        provider: The model provider to validate

    Returns:
        Tuple of (is_valid, message)
    """
    from docstra.core.config.settings import ModelProvider

    try:
        provider_enum = ModelProvider(provider.lower())
    except ValueError:
        return False, f"Invalid provider: {provider}"

    if provider_enum == ModelProvider.OLLAMA:
        # Check if Ollama is available, but don't fail hard
        try:
            from docstra.core.llm.ollama import OllamaClient

            client = OllamaClient(validate_connection=True)
            if client.connected:
                return True, "Ollama is available and ready"
            else:
                # Allow selection but warn user
                return True, (
                    "Warning: Ollama server not currently running. "
                    "Start with 'ollama serve' before using docstra commands"
                )
        except Exception as e:
            # Allow selection but warn user
            return True, (
                f"Warning: Could not verify Ollama connection: {e}. "
                "Ensure Ollama is installed and running before using docstra commands"
            )

    # For other providers, we assume they're valid (API keys will be validated later)
    return True, f"{provider_enum.value} provider selected"
