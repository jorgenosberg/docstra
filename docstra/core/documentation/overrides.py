# File: ./docstra/core/documentation/overrides.py
"""
Manual override system for documentation generation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DocumentationOverride:
    """Configuration for overriding documentation generation for a specific file."""

    file_path: str
    override_type: str  # "skip", "custom_template", "manual_content"
    custom_template: Optional[str] = None
    skip_generation: bool = False
    manual_content: Optional[str] = None
    created_at: float = 0.0
    created_by: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentationOverride:
        """Create from dictionary loaded from JSON."""
        return cls(**data)


class DocumentationOverrideManager:
    """Manages manual overrides for documentation generation."""

    def __init__(self, persist_directory: str):
        self.persist_directory = Path(persist_directory)
        self.overrides_file = self.persist_directory / "doc_overrides.json"
        self.overrides: Dict[str, DocumentationOverride] = {}

        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Load existing overrides
        self._load_overrides()

    def _load_overrides(self) -> None:
        """Load existing overrides from disk."""
        if self.overrides_file.exists():
            try:
                with open(self.overrides_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.overrides = {
                        path: DocumentationOverride.from_dict(override_data)
                        for path, override_data in data.items()
                    }
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Warning: Could not load overrides file: {e}")
                self.overrides = {}

    def _save_overrides(self) -> None:
        """Save overrides to disk."""
        try:
            data = {
                path: override.to_dict() for path, override in self.overrides.items()
            }
            with open(self.overrides_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save overrides: {e}")

    def set_file_override(
        self,
        file_path: str,
        override_type: str,
        custom_template: Optional[str] = None,
        skip_generation: bool = False,
        manual_content: Optional[str] = None,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> None:
        """Set manual override for a specific file's documentation.

        Args:
            file_path: Path to the file to override
            override_type: Type of override ("skip", "custom_template", "manual_content")
            custom_template: Custom template content for generation
            skip_generation: Whether to skip automatic generation
            manual_content: Manual content to use instead of generation
            description: Description of why this override exists
            created_by: Who created this override
        """
        # Normalize file path
        file_path = str(Path(file_path).resolve())

        self.overrides[file_path] = DocumentationOverride(
            file_path=file_path,
            override_type=override_type,
            custom_template=custom_template,
            skip_generation=skip_generation,
            manual_content=manual_content,
            created_at=time.time(),
            created_by=created_by,
            description=description,
        )
        self._save_overrides()

    def set_skip_override(
        self,
        file_path: str,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> None:
        """Skip documentation generation for a file."""
        self.set_file_override(
            file_path=file_path,
            override_type="skip",
            skip_generation=True,
            description=description,
            created_by=created_by,
        )

    def set_template_override(
        self,
        file_path: str,
        template_content: str,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> None:
        """Use custom template for a file's documentation."""
        self.set_file_override(
            file_path=file_path,
            override_type="custom_template",
            custom_template=template_content,
            description=description,
            created_by=created_by,
        )

    def set_manual_content_override(
        self,
        file_path: str,
        content: str,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> None:
        """Use manual content instead of generating documentation."""
        self.set_file_override(
            file_path=file_path,
            override_type="manual_content",
            manual_content=content,
            skip_generation=True,  # Skip generation since we have manual content
            description=description,
            created_by=created_by,
        )

    def remove_override(self, file_path: str) -> bool:
        """Remove override for a file.

        Returns:
            True if override was removed, False if it didn't exist
        """
        file_path = str(Path(file_path).resolve())

        if file_path in self.overrides:
            del self.overrides[file_path]
            self._save_overrides()
            return True

        return False

    def get_override(self, file_path: str) -> Optional[DocumentationOverride]:
        """Get override configuration for a file."""
        file_path = str(Path(file_path).resolve())
        return self.overrides.get(file_path)

    def should_skip_generation(self, file_path: str) -> bool:
        """Check if file should skip automatic generation."""
        override = self.get_override(file_path)
        return override is not None and override.skip_generation

    def get_custom_template(self, file_path: str) -> Optional[str]:
        """Get custom template for file if specified."""
        override = self.get_override(file_path)
        if override and override.override_type == "custom_template":
            return override.custom_template
        return None

    def get_manual_content(self, file_path: str) -> Optional[str]:
        """Get manual content for file if specified."""
        override = self.get_override(file_path)
        if override and override.override_type == "manual_content":
            return override.manual_content
        return None

    def list_overrides(
        self, override_type: Optional[str] = None
    ) -> List[DocumentationOverride]:
        """List all overrides, optionally filtered by type.

        Args:
            override_type: Filter by override type ("skip", "custom_template", "manual_content")

        Returns:
            List of overrides
        """
        overrides = list(self.overrides.values())

        if override_type:
            overrides = [o for o in overrides if o.override_type == override_type]

        # Sort by creation time (newest first)
        overrides.sort(key=lambda x: x.created_at, reverse=True)

        return overrides

    def get_override_stats(self) -> Dict[str, Any]:
        """Get statistics about active overrides."""
        if not self.overrides:
            return {"total": 0, "by_type": {}}

        by_type: Dict[str, int] = {}
        for override in self.overrides.values():
            override_type = override.override_type
            by_type[override_type] = by_type.get(override_type, 0) + 1

        return {
            "total": len(self.overrides),
            "by_type": by_type,
            "last_created": max(o.created_at for o in self.overrides.values())
            if self.overrides
            else 0,
        }

    def cleanup_missing_files(
        self, valid_file_paths: Optional[List[str]] = None
    ) -> int:
        """Remove overrides for files that no longer exist.

        Args:
            valid_file_paths: List of valid file paths. If None, check filesystem.

        Returns:
            Number of overrides removed
        """
        removed_count = 0
        to_remove = []

        for file_path, override in self.overrides.items():
            if valid_file_paths is not None:
                # Use provided list
                if file_path not in valid_file_paths:
                    to_remove.append(file_path)
            else:
                # Check filesystem
                if not Path(file_path).exists():
                    to_remove.append(file_path)

        for file_path in to_remove:
            del self.overrides[file_path]
            removed_count += 1

        if removed_count > 0:
            self._save_overrides()

        return removed_count

    def export_overrides(self, export_path: str) -> None:
        """Export overrides to a file for backup or sharing."""
        export_data = {
            "format_version": "1.0",
            "export_timestamp": time.time(),
            "overrides": {
                path: override.to_dict() for path, override in self.overrides.items()
            },
        }

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

    def import_overrides(self, import_path: str, merge: bool = True) -> int:
        """Import overrides from a file.

        Args:
            import_path: Path to the file to import from
            merge: If True, merge with existing overrides. If False, replace all.

        Returns:
            Number of overrides imported
        """
        with open(import_path, "r", encoding="utf-8") as f:
            import_data = json.load(f)

        if not merge:
            self.overrides.clear()

        imported_count = 0
        overrides_data = import_data.get("overrides", {})

        for path, override_data in overrides_data.items():
            try:
                override = DocumentationOverride.from_dict(override_data)
                self.overrides[path] = override
                imported_count += 1
            except Exception as e:
                print(f"Warning: Could not import override for {path}: {e}")

        if imported_count > 0:
            self._save_overrides()

        return imported_count

    def find_overrides_by_pattern(self, pattern: str) -> List[DocumentationOverride]:
        """Find overrides matching a file path pattern.

        Args:
            pattern: Glob-style pattern to match against file paths

        Returns:
            List of matching overrides
        """
        import fnmatch

        matching_overrides = []
        for file_path, override in self.overrides.items():
            if fnmatch.fnmatch(file_path, pattern):
                matching_overrides.append(override)

        return matching_overrides
