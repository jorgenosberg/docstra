"""Render stage: mkdocs configuration, navigation, and site build."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


def write_mkdocs_config(
    output_dir: Union[str, Path], site_name: str, site_description: str
) -> Path:
    """Write the mkdocs.yml configuration and return its path."""
    config = {
        "site_name": site_name,
        "site_description": site_description,
        "theme": {
            "name": "material",
            "palette": {"primary": "indigo", "accent": "indigo"},
            "features": [
                "navigation.instant",
                "navigation.tracking",
                "navigation.expand",
                "navigation.indexes",
                "search.highlight",
                "search.share",
                "toc.follow",
                "content.code.copy",
            ],
        },
        "markdown_extensions": [
            "pymdownx.highlight",
            "pymdownx.superfences",
            "pymdownx.inlinehilite",
            "pymdownx.tabbed",
            "admonition",
            "toc",
            "tables",
        ],
        "plugins": ["search"],
        "docs_dir": "docs",
    }

    config_path = Path(output_dir) / "mkdocs.yml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path


def update_navigation(output_dir: Union[str, Path]) -> None:
    """Build the nav section from the generated pages on disk."""
    output_dir = Path(output_dir)
    nav: List[Any] = [
        {"Home": "index.md"},
    ]

    guides_dir = output_dir / "docs" / "guides"
    if guides_dir.exists() and any(guides_dir.iterdir()):
        guide_items: List[Dict[str, str]] = []
        for guide_file in sorted(guides_dir.glob("*.md")):
            title = guide_file.stem.replace("-", " ").title()
            guide_items.append({title: f"guides/{guide_file.name}"})
        if guide_items:
            nav.append({"Guides": guide_items})

    modules_dir = output_dir / "docs" / "modules"
    if modules_dir.exists() and any(modules_dir.iterdir()):
        module_items: List[Dict[str, str]] = []
        for module_dir in sorted(modules_dir.iterdir()):
            if module_dir.is_dir() and (module_dir / "index.md").exists():
                title = module_dir.name.replace("_", " ").title()
                module_items.append({title: f"modules/{module_dir.name}/index.md"})
        if module_items:
            nav.append({"Modules": module_items})

    api_index = output_dir / "docs" / "api" / "index.md"
    if api_index.exists():
        nav.append({"API Reference": "api/index.md"})

    config_path = output_dir / "mkdocs.yml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["nav"] = nav
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)


def build_mkdocs_site(output_dir: Union[str, Path]) -> bool:
    """Run mkdocs build; return False when mkdocs is unavailable or fails."""
    try:
        subprocess.run(
            ["mkdocs", "build"],
            cwd=output_dir,
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
