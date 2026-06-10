import os
from typing import List

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table

from docstra.core.config.settings import ConfigManager, DocumentationConfig
from docstra.core.utils.colors import Colors


class DocumentationWizard:
    """Interactive wizard for documentation generation setup."""

    def __init__(self, console: Console, base_path: str, config_manager: ConfigManager):
        """Initialize the documentation wizard.

        Args:
            console: Rich console for UI
            base_path: Base path for the codebase
            config_manager: The configuration manager instance
        """
        self.console = console
        self.base_path = os.path.abspath(base_path)
        self.config_manager = config_manager

        # Load existing config from ConfigManager or use defaults
        existing_doc_config = self.config_manager.config.documentation
        if existing_doc_config:
            # Convert Pydantic model to dict for self.config, ensuring all keys exist
            self.config = existing_doc_config.model_dump()
        else:
            # Use defaults from DocumentationConfig Pydantic model
            default_doc_config = DocumentationConfig()
            self.config = default_doc_config.model_dump()

        # Ensure project name has a fallback if not in config
        if not self.config.get("project_name"):
            self.config["project_name"] = os.path.basename(self.base_path)

    def run(self) -> None:
        """Run the interactive wizard."""
        self.console.print(
            Panel(
                f"[{Colors.BOLD}]📚 Documentation Generation Wizard[/]",
                style=Colors.INFO_BOLD,
                expand=False,
            )
        )
        self.console.print(
            f"[{Colors.DIM}]Let's configure your documentation settings for a comprehensive codebase guide.[/]\n"
        )

        # Project information section
        self.console.print(f"[{Colors.BOLD}]📋 Project Information[/]")
        self.console.print("─" * 40)

        self._prompt_project_field(
            "project_name",
            "📝 Project Name",
            "The display name for your documentation site",
            self.config.get("project_name", ""),
        )

        self._prompt_project_field(
            "project_description",
            "📄 Project Description",
            "A brief description of what your project does",
            self.config.get("project_description", ""),
        )

        self._prompt_project_field(
            "project_version",
            "🏷️  Project Version",
            "Current version of your project (semantic versioning recommended)",
            self.config.get("project_version", "0.1.0"),
        )

        # Output configuration section
        self.console.print(f"\n[{Colors.BOLD}]⚙️  Output Configuration[/]")
        self.console.print("─" * 40)

        self._prompt_project_field(
            "output_dir",
            "📁 Output Directory",
            "Directory where documentation will be generated",
            self.config.get("output_dir", "./docs"),
        )

        formats = ["html", "markdown", "rst"]
        current_format = self.config.get("format", "markdown")
        format_idx = 0
        for i, fmt in enumerate(formats):
            if fmt == current_format:
                format_idx = i
                break

        self.console.print(f"\n[{Colors.HIGHLIGHT}]📄 Output Format[/]")
        self.console.print(
            f"[{Colors.DIM}]Choose the output format for your documentation[/]"
        )
        format_choice = Prompt.ask(
            "Select format", choices=formats, default=formats[format_idx]
        )
        self.config["format"] = format_choice

        # Directory selection
        self.console.print(f"\n[{Colors.BOLD}]🎯 Content Selection[/]")
        self.console.print("─" * 40)
        self._configure_directories()

        # Advanced options
        if Confirm.ask("🔧 Configure advanced options?", default=False):
            self.console.print(f"\n[{Colors.BOLD}]🔧 Advanced Options[/]")
            self.console.print("─" * 40)
            self._configure_advanced_options()

        # Summary with better formatting
        self.console.print(f"\n[{Colors.BOLD}]📊 Configuration Summary[/]")
        self.console.print("─" * 50)

        # Create a summary table with semantic styling
        summary_table = Table(show_header=False, box=None)
        summary_table.add_column("Setting", style=Colors.HIGHLIGHT, width=20)
        summary_table.add_column("Value", style=Colors.SUCCESS)

        # Add key settings to table
        key_settings = [
            ("📝 Project Name", self.config.get("project_name", "")),
            (
                "📄 Description",
                self.config.get("project_description", "")[:50]
                + (
                    "..."
                    if len(str(self.config.get("project_description", ""))) > 50
                    else ""
                ),
            ),
            ("🏷️  Version", self.config.get("project_version", "")),
            ("📁 Output Dir", self.config.get("output_dir", "")),
            ("📄 Format", self.config.get("format", "")),
            (
                "🎯 Include Dirs",
                f"{len(self.config.get('include_dirs', []))} specified"
                if self.config.get("include_dirs")
                else "All directories",
            ),
            (
                "🚫 Exclude Patterns",
                f"{len(self.config.get('exclude_patterns', []))} patterns"
                if self.config.get("exclude_patterns")
                else "None specified",
            ),
        ]

        for setting, value in key_settings:
            summary_table.add_row(
                setting, str(value) if value else f"[{Colors.DIM}]Not set[/]"
            )

        self.console.print(summary_table)
        self.console.print("─" * 50)

        # Save configuration with better messaging
        if Confirm.ask("💾 Save this configuration?", default=True):
            self._save_config()
        else:
            self.console.print(
                f"[{Colors.WARNING}]⚠️  Configuration not saved - using temporary settings for this run.[/]"
            )

    def _prompt_project_field(
        self, key: str, title: str, description: str, default: str
    ) -> None:
        """Prompt for a project configuration field with consistent styling."""
        self.console.print(f"\n[{Colors.HIGHLIGHT}]{title}[/]")
        self.console.print(f"[{Colors.DIM}]{description}[/]")

        value = Prompt.ask("Enter value", default=str(default) if default else "")
        self.config[key] = value

    def _configure_directories(self) -> None:
        """Configure included and excluded directories/patterns."""
        # First, discover directories
        available_dirs = self._discover_directories()

        # Show information about directory handling
        self.console.print(
            f"[{Colors.DIM}]ℹ️  Universal exclusions are managed in .docstra/.docstraignore[/]"
        )
        self.console.print(
            f"[{Colors.DIM}]   The following settings are specific to documentation generation.[/]\n"
        )

        if available_dirs:
            # Show available directories in a nice table
            dir_table = Table(title="Available Directories", show_header=True)
            dir_table.add_column("Directory", style=Colors.HIGHLIGHT)
            dir_table.add_column("Type", style=Colors.DIM)

            # Simple heuristic to identify directory types
            for dir_name in sorted(available_dirs[:10]):  # Show first 10
                dir_type = self._guess_directory_type(dir_name)
                dir_table.add_row(dir_name, dir_type)

            if len(available_dirs) > 10:
                dir_table.add_row(
                    f"[{Colors.DIM}]... and {len(available_dirs) - 10} more[/]", ""
                )

            self.console.print(dir_table)

        # Documentation-specific exclusions
        self.console.print(
            f"\n[{Colors.HIGHLIGHT}]🚫 Documentation Exclude Patterns[/]"
        )
        self.console.print(
            f"[{Colors.DIM}]Additional gitignore-style patterns to exclude from documentation (e.g., 'tests/*', '*.tmp')[/]"
        )

        current_excludes = self.config.get("exclude_patterns") or []
        exclude_input = Prompt.ask(
            "Exclude patterns (comma-separated)",
            default=",".join(current_excludes) if current_excludes else "",
        )
        if exclude_input.strip():
            self.config["exclude_patterns"] = [
                p.strip() for p in exclude_input.split(",") if p.strip()
            ]
        else:
            self.config["exclude_patterns"] = []

        # Optional directory inclusion
        if Confirm.ask(
            "🎯 Specify specific directories to include?",
            default=False,
        ):
            self.console.print(
                f"[{Colors.DIM}]Leave empty to include all directories (recommended for comprehensive docs)[/]"
            )
            current_includes = self.config.get("include_dirs") or []
            include_input = Prompt.ask(
                "Include directories (comma-separated)",
                default=",".join(current_includes) if current_includes else "",
            )
            if include_input.strip():
                self.config["include_dirs"] = [
                    d.strip() for d in include_input.split(",") if d.strip()
                ]
            else:
                self.config["include_dirs"] = []
        else:
            self.config["include_dirs"] = []

    def _discover_directories(self) -> List[str]:
        """Discover directories in the base path.

        Returns:
            List of directory names
        """
        try:
            return [
                d
                for d in os.listdir(self.base_path)
                if os.path.isdir(os.path.join(self.base_path, d))
                and not d.startswith(".")
            ]
        except Exception as e:
            self.console.print(
                f"[{Colors.ERROR_BOLD}]Error discovering directories:[/] {str(e)}"
            )
            return []

    def _guess_directory_type(self, dir_name: str) -> str:
        """Guess the type of directory based on common naming patterns."""
        dir_name_lower = dir_name.lower()

        if dir_name_lower in ["src", "source", "lib", "library"]:
            return "📦 Source Code"
        elif dir_name_lower in ["test", "tests", "testing", "spec", "specs"]:
            return "🧪 Tests"
        elif dir_name_lower in ["doc", "docs", "documentation"]:
            return "📚 Documentation"
        elif dir_name_lower in ["example", "examples", "demo", "demos"]:
            return "💡 Examples"
        elif dir_name_lower in ["config", "configuration", "settings"]:
            return "⚙️  Configuration"
        elif dir_name_lower in ["build", "dist", "target", "out", "output"]:
            return "🏗️  Build Output"
        elif dir_name_lower in ["node_modules", "vendor", "third_party"]:
            return "📦 Dependencies"
        else:
            return "📁 Directory"

    def _configure_advanced_options(self) -> None:
        """Configure advanced documentation options."""
        # Theme selection
        themes = ["default", "readthedocs", "material", "sphinx_rtd_theme"]
        current_theme = self.config.get("theme", "default")
        theme_idx = 0
        for i, theme in enumerate(themes):
            if theme == current_theme:
                theme_idx = i
                break

        self.console.print(f"\n[{Colors.HIGHLIGHT}]🎨 Documentation Theme[/]")
        self.console.print(f"[{Colors.DIM}]Visual theme for your documentation site[/]")
        theme_choice = Prompt.ask(
            "Select theme", choices=themes, default=themes[theme_idx]
        )
        self.config["theme"] = theme_choice

    def _save_config(self) -> None:
        """Save configuration to the ConfigManager."""
        try:
            # Update the documentation section of the main config
            if self.config_manager.config.documentation is None:
                self.config_manager.config.documentation = DocumentationConfig(
                    include_dirs=self.config.get("include_dirs"),
                    exclude_patterns=self.config.get("exclude_patterns"),
                    output_dir=self.config.get("output_dir", "./docs"),
                    format=self.config.get("format", "markdown"),
                    theme=self.config.get("theme", "default"),
                    project_name=self.config.get("project_name"),
                    project_description=self.config.get("project_description"),
                    project_version=self.config.get("project_version", "0.1.0"),
                    documentation_structure=self.config.get(
                        "documentation_structure", "file_based"
                    ),
                    module_doc_depth=self.config.get("module_doc_depth", "full"),
                    llm_style_prompt=self.config.get("llm_style_prompt"),
                    max_workers_ollama=self.config.get("max_workers_ollama", 1),
                    max_workers_api=self.config.get("max_workers_api", 4),
                    max_workers_default=self.config.get("max_workers_default"),
                )
            else:
                for key, value in self.config.items():
                    if hasattr(self.config_manager.config.documentation, key):
                        setattr(self.config_manager.config.documentation, key, value)
                    else:
                        # This case should ideally not happen if self.config is derived from DocumentationConfig model_dump
                        self.console.print(
                            f"[{Colors.WARNING}]Warning: Unknown key '{key}' found in wizard config. Skipping.[/]"
                        )

            self.config_manager.save()
            self.console.print(
                f"[{Colors.SUCCESS_BOLD}]✅ Configuration saved successfully![/]"
            )
            self.console.print(
                f"[{Colors.DIM}]📍 Location: {self.config_manager.config_path}[/]"
            )
        except Exception as e:
            self.console.print(
                f"[{Colors.ERROR_BOLD}]❌ Error saving configuration:[/] {str(e)}"
            )
            self.console.print(
                f"[{Colors.WARNING}]⚠️  Please check your configuration file: {self.config_manager.config_path}[/]"
            )


def run_documentation_wizard(
    console: Console, path: str, config_manager: ConfigManager
) -> None:
    """Run the documentation wizard."""
    wizard = DocumentationWizard(console, path, config_manager)
    wizard.run()
