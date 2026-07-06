"""Tests for default directory exclusions in the file collector."""

from __future__ import annotations

from pathlib import Path

from docstra.core.utils.file_collector import FileCollector, collect_files


def _make_tree(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("def main(): ...\n")
    (repo / ".venv" / "lib" / "site-packages" / "torch").mkdir(parents=True)
    (repo / ".venv" / "lib" / "site-packages" / "torch" / "core.py").write_text("x=1\n")
    (repo / "node_modules" / "pkg").mkdir(parents=True)
    (repo / "node_modules" / "pkg" / "index.js").write_text("var x=1\n")
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "app.py").write_text("cached\n")
    (repo / "dist").mkdir()
    (repo / "dist" / "bundle.js").write_text("var y=1\n")
    (repo / ".git").mkdir()
    (repo / ".git" / "config.ini").write_text("[core]\n")
    return repo


def test_dependency_and_cache_directories_are_excluded_by_default(
    tmp_path: Path,
) -> None:
    repo = _make_tree(tmp_path)

    files = collect_files(base_path=str(repo), file_extensions=[".py", ".js", ".ini"])

    assert [str(f.relative_to(repo)) for f in files] == ["src/app.py"]


def test_default_excludes_prune_without_walking(tmp_path: Path) -> None:
    repo = _make_tree(tmp_path)

    collector = FileCollector(base_path=str(repo), file_extensions=[".py", ".js"])
    collector.collect_files()

    # .venv is pruned at the top, so its subtree is never visited
    assert collector.stats["visited_files"] < 6


def test_explicit_include_overrides_default_excludes(tmp_path: Path) -> None:
    repo = _make_tree(tmp_path)

    collector = FileCollector(
        base_path=str(repo), include_dirs=[".venv"], file_extensions=[".py"]
    )
    files = collector.collect_files()

    assert ".venv/lib/site-packages/torch/core.py" in {
        str(f.relative_to(repo)) for f in files
    }


def test_default_excludes_can_be_disabled(tmp_path: Path) -> None:
    repo = _make_tree(tmp_path)

    collector = FileCollector(
        base_path=str(repo), file_extensions=[".py"], use_default_excludes=False
    )
    files = {str(f.relative_to(repo)) for f in collector.collect_files()}

    assert "src/app.py" in files
    assert ".venv/lib/site-packages/torch/core.py" in files
