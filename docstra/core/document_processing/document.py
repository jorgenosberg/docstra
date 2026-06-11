# File: ./docstra/core/document_processing/document.py
"""
Document models for representing code documents and their metadata.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Types of code documents that can be processed."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    MARKDOWN = "markdown"
    TEXT = "text"
    OTHER = "other"

    def __str__(self) -> str:
        """Return the enum value instead of the enum representation."""
        return self.value


class DocumentMetadata(BaseModel):
    """Metadata for a document."""

    filepath: str = Field(..., description="Path to the document")
    language: DocumentType = Field(
        ..., description="Programming language of the document"
    )
    size_bytes: int = Field(..., description="Size of the document in bytes")
    last_modified: float = Field(..., description="Last modified timestamp")
    line_count: int = Field(0, description="Number of lines in the document")
    imports: List[str] = Field(
        default_factory=list, description="Imports used in the document"
    )
    classes: List[str] = Field(
        default_factory=list, description="Classes defined in the document"
    )
    functions: List[str] = Field(
        default_factory=list, description="Functions defined in the document"
    )
    symbols: Dict[str, List[int]] = Field(
        default_factory=dict, description="Symbol to line numbers mapping"
    )
    module_docstring: Optional[str] = Field(
        None, description="Module level docstring if present"
    )

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> DocumentMetadata:
        """Create metadata from a file path."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File {filepath} not found")

        size = path.stat().st_size
        mtime = path.stat().st_mtime

        # Determine language from file extension
        extension = path.suffix.lower()
        language = DocumentType.OTHER
        if extension == ".py":
            language = DocumentType.PYTHON
        elif extension in [".js", ".mjs", ".cjs", ".jsx"]:
            language = DocumentType.JAVASCRIPT
        elif extension in [".ts", ".tsx"]:
            language = DocumentType.TYPESCRIPT
        elif extension == ".java":
            language = DocumentType.JAVA
        elif extension == ".go":
            language = DocumentType.GO
        elif extension == ".rs":
            language = DocumentType.RUST
        elif extension in [".cpp", ".cc", ".cxx"]:
            language = DocumentType.CPP
        elif extension == ".c":
            language = DocumentType.C
        elif extension == ".cs":
            language = DocumentType.CSHARP
        elif extension == ".php":
            language = DocumentType.PHP
        elif extension == ".rb":
            language = DocumentType.RUBY
        elif extension == ".swift":
            language = DocumentType.SWIFT
        elif extension == ".kt":
            language = DocumentType.KOTLIN
        elif extension == ".md":
            language = DocumentType.MARKDOWN
        elif extension == ".txt":
            language = DocumentType.TEXT

        # Basic line count (will be enhanced by parser)
        line_count = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))

        return cls(
            filepath=str(path.absolute()),
            language=language,
            size_bytes=size,
            last_modified=mtime,
            line_count=line_count,
            module_docstring=None,
        )


class CodeChunk(BaseModel):
    """A chunk of code with its context."""

    content: str = Field(..., description="The content of the chunk")
    start_line: int = Field(..., description="Start line of the chunk")
    end_line: int = Field(..., description="End line of the chunk")
    symbols: List[str] = Field(
        default_factory=list, description="Symbols in this chunk"
    )
    chunk_type: str = Field(
        "code", description="Type of the chunk (function, class, etc.)"
    )
    parent_symbols: List[str] = Field(
        default_factory=list, description="Parent symbols (containing class/function)"
    )


class Document(BaseModel):
    """A code document with its content and metadata."""

    content: str = Field(..., description="The content of the document")
    metadata: DocumentMetadata = Field(..., description="Metadata of the document")
    chunks: List[CodeChunk] = Field(
        default_factory=list, description="Chunks of the document"
    )
    embedding_id: Optional[str] = Field(
        None, description="ID in the embedding database"
    )

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> Document:
        """Create a document from a file path."""
        path = Path(filepath)
        metadata = DocumentMetadata.from_file(path)

        # Read file content with error handling
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            content = f"Error reading file: {str(e)}"

        return cls(content=content, metadata=metadata, embedding_id=str(uuid.uuid4()))
