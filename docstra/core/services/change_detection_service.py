# File: ./docstra/core/services/change_detection_service.py
"""
Change detection service for incremental documentation updates.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from docstra.core.utils.file_collector import FileCollector


@dataclass
class ChangeAnalysis:
    """Analysis of changes detected in the codebase."""

    changed_files: List[str]
    new_files: List[str]
    deleted_files: List[str]
    total_files: int
    change_timestamp: float
    git_based: bool = False
    base_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return (
            len(self.changed_files) > 0
            or len(self.new_files) > 0
            or len(self.deleted_files) > 0
        )

    @property
    def total_changes(self) -> int:
        """Total number of changed files."""
        return len(self.changed_files) + len(self.new_files) + len(self.deleted_files)


class ChangeDetectionService:
    """Detects changes in source files and determines documentation impact."""

    def __init__(self, persist_directory: str):
        self.persist_directory = Path(persist_directory)
        self.change_log_file = self.persist_directory / "change_log.json"
        self.file_hashes_file = self.persist_directory / "file_hashes.json"
        self.last_generation_file = self.persist_directory / "last_generation.json"

        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

    def detect_changes_since_last_generation(
        self, codebase_path: str
    ) -> ChangeAnalysis:
        """Detect all changes since last documentation generation."""
        current_hashes = self._calculate_file_hashes(codebase_path)
        previous_hashes = self._load_previous_hashes()

        changed_files = []
        new_files = []
        deleted_files = []

        # Find changed and new files
        for file_path, current_hash in current_hashes.items():
            if file_path not in previous_hashes:
                new_files.append(file_path)
            elif previous_hashes[file_path] != current_hash:
                changed_files.append(file_path)

        # Find deleted files
        for file_path in previous_hashes:
            if file_path not in current_hashes:
                deleted_files.append(file_path)

        change_analysis = ChangeAnalysis(
            changed_files=changed_files,
            new_files=new_files,
            deleted_files=deleted_files,
            total_files=len(current_hashes),
            change_timestamp=time.time(),
        )

        # Log the change analysis
        self._log_change_analysis(change_analysis)

        return change_analysis

    def detect_changes_from_git(
        self, codebase_path: str, base_ref: str = "HEAD~1"
    ) -> ChangeAnalysis:
        """Detect changes using Git for CI/CD integration."""
        try:
            # Get list of changed files from Git
            changed_result = subprocess.run(
                ["git", "diff", "--name-only", base_ref, "HEAD"],
                cwd=codebase_path,
                capture_output=True,
                text=True,
                check=True,
            )

            # Get list of new files (added in this commit)
            new_result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=A", base_ref, "HEAD"],
                cwd=codebase_path,
                capture_output=True,
                text=True,
                check=True,
            )

            # Get list of deleted files
            deleted_result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=D", base_ref, "HEAD"],
                cwd=codebase_path,
                capture_output=True,
                text=True,
                check=True,
            )

            codebase_path_obj = Path(codebase_path).resolve()

            # Filter and resolve file paths
            changed_files = [
                str(codebase_path_obj / f.strip())
                for f in changed_result.stdout.split("\n")
                if f.strip() and self._is_documentable_file(f.strip())
            ]

            new_files = [
                str(codebase_path_obj / f.strip())
                for f in new_result.stdout.split("\n")
                if f.strip() and self._is_documentable_file(f.strip())
            ]

            deleted_files = [
                str(codebase_path_obj / f.strip())
                for f in deleted_result.stdout.split("\n")
                if f.strip() and self._is_documentable_file(f.strip())
            ]

            # Remove new files from changed files (avoid duplication)
            changed_files = [f for f in changed_files if f not in new_files]

            change_analysis = ChangeAnalysis(
                changed_files=changed_files,
                new_files=new_files,
                deleted_files=deleted_files,
                total_files=len(changed_files) + len(new_files),
                change_timestamp=time.time(),
                git_based=True,
                base_ref=base_ref,
            )

            self._log_change_analysis(change_analysis)
            return change_analysis

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Git detection failed ({e}), falling back to filesystem detection")
            # Fallback to file-based detection
            return self.detect_changes_since_last_generation(codebase_path)

    def detect_changes_from_file_list(self, file_paths: List[str]) -> ChangeAnalysis:
        """Detect changes from an explicit list of files (for manual triggers)."""
        existing_files = []
        missing_files = []

        for file_path in file_paths:
            if Path(file_path).exists():
                existing_files.append(file_path)
            else:
                missing_files.append(file_path)

        return ChangeAnalysis(
            changed_files=existing_files,
            new_files=[],
            deleted_files=missing_files,
            total_files=len(existing_files),
            change_timestamp=time.time(),
        )

    def mark_generation_complete(self, codebase_path: str) -> None:
        """Mark that documentation generation is complete and update hashes."""
        # Update file hashes
        current_hashes = self._calculate_file_hashes(codebase_path)
        self._save_file_hashes(current_hashes)

        # Update last generation timestamp
        generation_info = {
            "timestamp": time.time(),
            "total_files": len(current_hashes),
            "codebase_path": str(Path(codebase_path).resolve()),
        }

        try:
            with open(self.last_generation_file, "w", encoding="utf-8") as f:
                json.dump(generation_info, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save generation info: {e}")

    def get_last_generation_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the last documentation generation."""
        if not self.last_generation_file.exists():
            return None

        try:
            with open(self.last_generation_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _calculate_file_hashes(self, codebase_path: str) -> Dict[str, str]:
        """Calculate hashes for all documentable files in the codebase."""
        file_hashes = {}
        codebase_path_obj = Path(codebase_path).resolve()

        # Get all documentable files
        try:
            from docstra.core.utils.file_collector import collect_files

            files = collect_files(
                base_path=str(codebase_path_obj),
                file_extensions=FileCollector.default_code_file_extensions(),
            )
        except Exception:
            # Fallback to manual discovery
            files = []
            for ext in [".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".h"]:
                files.extend(codebase_path_obj.rglob(f"*{ext}"))

        for file_path in files:
            try:
                file_path_str = str(file_path)
                file_hash = self._calculate_file_hash(file_path_str)
                if file_hash:
                    file_hashes[file_path_str] = file_hash
            except Exception:
                # Skip files that can't be processed
                continue

        return file_hashes

    def _calculate_file_hash(self, file_path: str) -> Optional[str]:
        """Calculate hash for a single file."""
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return None

            # Use both content and modification time for hash
            with open(file_path_obj, "rb") as f:
                content = f.read()

            mtime = file_path_obj.stat().st_mtime
            hash_input = content + str(mtime).encode("utf-8")

            return hashlib.sha256(hash_input).hexdigest()
        except Exception:
            return None

    def _load_previous_hashes(self) -> Dict[str, str]:
        """Load previously calculated file hashes."""
        if not self.file_hashes_file.exists():
            return {}

        try:
            with open(self.file_hashes_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_file_hashes(self, file_hashes: Dict[str, str]) -> None:
        """Save file hashes to disk."""
        try:
            with open(self.file_hashes_file, "w", encoding="utf-8") as f:
                json.dump(file_hashes, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save file hashes: {e}")

    def _log_change_analysis(self, analysis: ChangeAnalysis) -> None:
        """Log change analysis for debugging and audit purposes."""
        try:
            # Load existing log or create new one
            log_entries = []
            if self.change_log_file.exists():
                try:
                    with open(self.change_log_file, "r", encoding="utf-8") as f:
                        log_entries = json.load(f)
                except Exception:
                    log_entries = []

            # Add new entry
            log_entries.append(analysis.to_dict())

            # Keep only last 50 entries
            log_entries = log_entries[-50:]

            # Save log
            with open(self.change_log_file, "w", encoding="utf-8") as f:
                json.dump(log_entries, f, indent=2, default=str)

        except Exception as e:
            print(f"Warning: Could not log change analysis: {e}")

    def _is_documentable_file(self, file_path: str) -> bool:
        """Check if a file should be included in documentation."""
        file_path_obj = Path(file_path)

        # Check extension
        documentable_extensions = FileCollector.default_code_file_extensions()
        if file_path_obj.suffix.lower() not in documentable_extensions:
            return False

        # Check for common exclude patterns
        exclude_patterns = [
            "__pycache__",
            ".pyc",
            ".git",
            "node_modules",
            ".mypy_cache",
            ".pytest_cache",
            "venv",
            ".venv",
            "build",
            "dist",
            "target",
            ".tox",
            ".nox",
        ]

        path_str = str(file_path_obj)
        for pattern in exclude_patterns:
            if pattern in path_str:
                return False

        return True

    def get_change_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent change detection history."""
        if not self.change_log_file.exists():
            return []

        try:
            with open(self.change_log_file, "r", encoding="utf-8") as f:
                log_entries = json.load(f)
                return log_entries[-limit:] if log_entries else []
        except Exception:
            return []
