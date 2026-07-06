"""Docstra: A tool for semantic code search and documentation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("docstra")
except PackageNotFoundError:
    __version__ = "0.3.1"
