# File: ./docstra/core/document_processing/parser.py
"""
Code parser using Tree-sitter for extracting structure and metadata from code files.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, cast

import tree_sitter
from tree_sitter import Language, Parser, Tree
from tree_sitter_language_pack import get_language, get_parser

from docstra.core.document_processing.document import (
    CodeChunk,
    Document,
    DocumentMetadata,
    DocumentType,
)


class CodeParser:
    """Parser for code files using Tree-sitter."""

    # Language to parser mapping
    LANGUAGES: Dict[DocumentType, str] = {
        DocumentType.PYTHON: "python",
        DocumentType.JAVASCRIPT: "javascript",
        DocumentType.TYPESCRIPT: "typescript",
        DocumentType.JAVA: "java",
        DocumentType.GO: "go",
        DocumentType.RUST: "rust",
        DocumentType.CPP: "cpp",
        DocumentType.C: "c",
        DocumentType.CSHARP: "c_sharp",
        DocumentType.PHP: "php",
        DocumentType.RUBY: "ruby",
    }

    def __init__(self, languages_dir: Optional[str] = None) -> None:
        """Initialize the parser with Tree-sitter languages.

        Args:
            languages_dir: Directory containing Tree-sitter language libraries.
                If None, will attempt to download and build languages.
        """
        self._parsers: Dict[DocumentType, Parser] = {}
        self._languages: Dict[DocumentType, Language] = {}

        # Check which languages are available in the language pack
        self._available_languages = set()
        try:
            for lang_name in self.LANGUAGES.values():
                try:
                    # Check if we can get this language
                    get_language(
                        cast(
                            Literal[
                                "actionscript",
                                "ada",
                                "agda",
                                "arduino",
                                "asm",
                                "astro",
                                "bash",
                                "beancount",
                                "bibtex",
                                "bicep",
                                "bitbake",
                                "c",
                                "cairo",
                                "capnp",
                                "chatito",
                                "clarity",
                                "clojure",
                                "cmake",
                                "comment",
                                "commonlisp",
                                "cpon",
                                "cpp",
                                "csharp",
                                "css",
                                "csv",
                                "cuda",
                                "d",
                                "dart",
                                "dockerfile",
                                "doxygen",
                                "dtd",
                                "elisp",
                                "elixir",
                                "elm",
                                "embeddedtemplate",
                                "erlang",
                                "fennel",
                                "firrtl",
                                "fish",
                                "fortran",
                                "func",
                                "gdscript",
                                "gitattributes",
                                "gitcommit",
                                "gitignore",
                                "gleam",
                                "glsl",
                                "gn",
                                "go",
                                "gomod",
                                "gosum",
                                "groovy",
                                "gstlaunch",
                                "hack",
                                "hare",
                                "haskell",
                                "haxe",
                                "hcl",
                                "heex",
                                "hlsl",
                                "html",
                                "hyprlang",
                                "ispc",
                                "janet",
                                "java",
                                "javascript",
                                "jsdoc",
                                "json",
                                "jsonnet",
                                "julia",
                                "kconfig",
                                "kdl",
                                "kotlin",
                                "latex",
                                "linkerscript",
                                "llvm",
                                "lua",
                                "luadoc",
                                "luap",
                                "luau",
                                "make",
                                "markdown",
                                "matlab",
                                "mermaid",
                                "meson",
                                "ninja",
                                "nix",
                                "nqc",
                                "objc",
                                "odin",
                                "org",
                                "pascal",
                                "pem",
                                "perl",
                                "pgn",
                                "php",
                                "po",
                                "pony",
                                "powershell",
                                "printf",
                                "prisma",
                                "properties",
                                "proto",
                                "psv",
                                "puppet",
                                "purescript",
                                "pymanifest",
                                "python",
                                "qmldir",
                                "qmljs",
                                "query",
                                "r",
                                "racket",
                                "re2c",
                                "readline",
                                "requirements",
                                "ron",
                                "rst",
                                "ruby",
                                "rust",
                                "scala",
                                "scheme",
                                "scss",
                                "smali",
                                "smithy",
                                "solidity",
                                "sparql",
                                "swift",
                                "sql",
                                "squirrel",
                                "starlark",
                                "svelte",
                                "tablegen",
                                "tcl",
                                "terraform",
                                "test",
                                "thrift",
                                "toml",
                                "tsv",
                                "tsx",
                                "twig",
                                "typescript",
                                "typst",
                                "udev",
                                "ungrammar",
                                "uxntal",
                                "v",
                                "verilog",
                                "vhdl",
                                "vim",
                                "vue",
                                "wgsl",
                                "xcompose",
                                "xml",
                                "yaml",
                                "yuck",
                                "zig",
                                "magik",
                            ],
                            lang_name,
                        )
                    )
                    self._available_languages.add(lang_name)
                except Exception:
                    # Language not available in the pack
                    pass
        except Exception as e:
            print(f"Warning: tree_sitter_language_pack not fully accessible: {str(e)}")

    def parse_document(self, document: Document) -> Document:
        """Parse a document to extract structure and metadata.

        Args:
            document: The document to parse

        Returns:
            The document with updated metadata and chunks
        """
        # Get parser for this document's language
        parser = self._get_parser_for_language(document.metadata.language)

        if not parser:
            # Skip parsing for unsupported languages
            return document

        # Parse the document
        tree = parser.parse(bytes(document.content, "utf-8"))

        # Extract metadata and chunks
        metadata = self._extract_metadata(tree, document.metadata)
        chunks = self._extract_chunks(tree, document.content)

        # Update the document
        document.metadata = metadata
        document.chunks = chunks

        return document

    def _get_parser_for_language(self, doc_type: DocumentType) -> Optional[Parser]:
        """Get a parser for the specified language type.

        Args:
            doc_type: Document type to get parser for

        Returns:
            Parser for the language, or None if not available
        """
        # Return cached parser if available
        if doc_type in self._parsers:
            return self._parsers[doc_type]

        # Get language name from mapping
        lang_name = self.LANGUAGES.get(doc_type)
        if not lang_name or lang_name not in self._available_languages:
            return None

        try:
            # Get parser from language pack
            parser = get_parser(
                cast(
                    Literal[
                        "actionscript",
                        "ada",
                        "agda",
                        "arduino",
                        "asm",
                        "astro",
                        "bash",
                        "beancount",
                        "bibtex",
                        "bicep",
                        "bitbake",
                        "c",
                        "cairo",
                        "capnp",
                        "chatito",
                        "clarity",
                        "clojure",
                        "cmake",
                        "comment",
                        "commonlisp",
                        "cpon",
                        "cpp",
                        "csharp",
                        "css",
                        "csv",
                        "cuda",
                        "d",
                        "dart",
                        "dockerfile",
                        "doxygen",
                        "dtd",
                        "elisp",
                        "elixir",
                        "elm",
                        "embeddedtemplate",
                        "erlang",
                        "fennel",
                        "firrtl",
                        "fish",
                        "fortran",
                        "func",
                        "gdscript",
                        "gitattributes",
                        "gitcommit",
                        "gitignore",
                        "gleam",
                        "glsl",
                        "gn",
                        "go",
                        "gomod",
                        "gosum",
                        "groovy",
                        "gstlaunch",
                        "hack",
                        "hare",
                        "haskell",
                        "haxe",
                        "hcl",
                        "heex",
                        "hlsl",
                        "html",
                        "hyprlang",
                        "ispc",
                        "janet",
                        "java",
                        "javascript",
                        "jsdoc",
                        "json",
                        "jsonnet",
                        "julia",
                        "kconfig",
                        "kdl",
                        "kotlin",
                        "latex",
                        "linkerscript",
                        "llvm",
                        "lua",
                        "luadoc",
                        "luap",
                        "luau",
                        "make",
                        "markdown",
                        "matlab",
                        "mermaid",
                        "meson",
                        "ninja",
                        "nix",
                        "nqc",
                        "objc",
                        "odin",
                        "org",
                        "pascal",
                        "pem",
                        "perl",
                        "pgn",
                        "php",
                        "po",
                        "pony",
                        "powershell",
                        "printf",
                        "prisma",
                        "properties",
                        "proto",
                        "psv",
                        "puppet",
                        "purescript",
                        "pymanifest",
                        "python",
                        "qmldir",
                        "qmljs",
                        "query",
                        "r",
                        "racket",
                        "re2c",
                        "readline",
                        "requirements",
                        "ron",
                        "rst",
                        "ruby",
                        "rust",
                        "scala",
                        "scheme",
                        "scss",
                        "smali",
                        "smithy",
                        "solidity",
                        "sparql",
                        "swift",
                        "sql",
                        "squirrel",
                        "starlark",
                        "svelte",
                        "tablegen",
                        "tcl",
                        "terraform",
                        "test",
                        "thrift",
                        "toml",
                        "tsv",
                        "tsx",
                        "twig",
                        "typescript",
                        "typst",
                        "udev",
                        "ungrammar",
                        "uxntal",
                        "v",
                        "verilog",
                        "vhdl",
                        "vim",
                        "vue",
                        "wgsl",
                        "xcompose",
                        "xml",
                        "yaml",
                        "yuck",
                        "zig",
                        "magik",
                    ],
                    lang_name,
                )
            )

            # Cache the parser and language for future use
            self._parsers[doc_type] = parser
            self._languages[doc_type] = get_language(
                cast(
                    Literal[
                        "actionscript",
                        "ada",
                        "agda",
                        "arduino",
                        "asm",
                        "astro",
                        "bash",
                        "beancount",
                        "bibtex",
                        "bicep",
                        "bitbake",
                        "c",
                        "cairo",
                        "capnp",
                        "chatito",
                        "clarity",
                        "clojure",
                        "cmake",
                        "comment",
                        "commonlisp",
                        "cpon",
                        "cpp",
                        "csharp",
                        "css",
                        "csv",
                        "cuda",
                        "d",
                        "dart",
                        "dockerfile",
                        "doxygen",
                        "dtd",
                        "elisp",
                        "elixir",
                        "elm",
                        "embeddedtemplate",
                        "erlang",
                        "fennel",
                        "firrtl",
                        "fish",
                        "fortran",
                        "func",
                        "gdscript",
                        "gitattributes",
                        "gitcommit",
                        "gitignore",
                        "gleam",
                        "glsl",
                        "gn",
                        "go",
                        "gomod",
                        "gosum",
                        "groovy",
                        "gstlaunch",
                        "hack",
                        "hare",
                        "haskell",
                        "haxe",
                        "hcl",
                        "heex",
                        "hlsl",
                        "html",
                        "hyprlang",
                        "ispc",
                        "janet",
                        "java",
                        "javascript",
                        "jsdoc",
                        "json",
                        "jsonnet",
                        "julia",
                        "kconfig",
                        "kdl",
                        "kotlin",
                        "latex",
                        "linkerscript",
                        "llvm",
                        "lua",
                        "luadoc",
                        "luap",
                        "luau",
                        "make",
                        "markdown",
                        "matlab",
                        "mermaid",
                        "meson",
                        "ninja",
                        "nix",
                        "nqc",
                        "objc",
                        "odin",
                        "org",
                        "pascal",
                        "pem",
                        "perl",
                        "pgn",
                        "php",
                        "po",
                        "pony",
                        "powershell",
                        "printf",
                        "prisma",
                        "properties",
                        "proto",
                        "psv",
                        "puppet",
                        "purescript",
                        "pymanifest",
                        "python",
                        "qmldir",
                        "qmljs",
                        "query",
                        "r",
                        "racket",
                        "re2c",
                        "readline",
                        "requirements",
                        "ron",
                        "rst",
                        "ruby",
                        "rust",
                        "scala",
                        "scheme",
                        "scss",
                        "smali",
                        "smithy",
                        "solidity",
                        "sparql",
                        "swift",
                        "sql",
                        "squirrel",
                        "starlark",
                        "svelte",
                        "tablegen",
                        "tcl",
                        "terraform",
                        "test",
                        "thrift",
                        "toml",
                        "tsv",
                        "tsx",
                        "twig",
                        "typescript",
                        "typst",
                        "udev",
                        "ungrammar",
                        "uxntal",
                        "v",
                        "verilog",
                        "vhdl",
                        "vim",
                        "vue",
                        "wgsl",
                        "xcompose",
                        "xml",
                        "yaml",
                        "yuck",
                        "zig",
                        "magik",
                    ],
                    lang_name,
                )
            )

            return parser
        except Exception as e:
            print(f"Error loading parser for {lang_name}: {str(e)}")
            return None

    def _extract_metadata(
        self, tree: Tree, metadata: DocumentMetadata
    ) -> DocumentMetadata:
        """Extract metadata from a parsed tree.

        Args:
            tree: The parsed tree
            metadata: The existing metadata to update

        Returns:
            Updated metadata
        """
        root_node = tree.root_node

        # Extract imports, classes, and functions based on language
        if metadata.language == DocumentType.PYTHON:
            metadata.imports = self._extract_python_imports(root_node)
            metadata.classes = self._extract_python_classes(root_node)
            metadata.functions = self._extract_python_functions(root_node)
            metadata.symbols = self._build_symbol_table(
                root_node, ["class_definition", "function_definition"]
            )

        elif metadata.language in [DocumentType.JAVASCRIPT, DocumentType.TYPESCRIPT]:
            metadata.imports = self._extract_js_imports(root_node)
            metadata.classes = self._extract_js_classes(root_node)
            metadata.functions = self._extract_js_functions(root_node)
            metadata.symbols = self._build_symbol_table(
                root_node,
                ["class_declaration", "function_declaration", "method_definition"],
            )

        # Add more language-specific extractors as needed

        return metadata

    def _extract_chunks(self, tree: Tree, content: str) -> List[CodeChunk]:
        """Extract code chunks from a parsed tree.

        Args:
            tree: The parsed tree
            content: The document content

        Returns:
            List of code chunks
        """
        chunks: List[CodeChunk] = []
        root_node = tree.root_node

        # Find chunking points (functions, classes, etc.)
        chunking_nodes: List[tree_sitter.Node] = []
        self._collect_nodes_by_type(
            root_node,
            [
                "function_definition",
                "class_definition",
                "method_definition",
                "function_declaration",
                "class_declaration",
            ],
            chunking_nodes,
        )

        # Create chunks from nodes
        for node in chunking_nodes:
            chunk_content = content[node.start_byte : node.end_byte]
            start_line = node.start_point[0] + 1  # Tree-sitter is 0-indexed
            end_line = node.end_point[0] + 1

            # Determine chunk type and name
            chunk_type = node.type.replace("_definition", "").replace(
                "_declaration", ""
            )
            name_node = self._find_name_node(node)
            symbol = (
                self.safe_decode(name_node.text)
                if name_node and name_node.text is not None
                else None
            )
            symbols = [symbol] if symbol is not None else []

            # Find parent symbols
            parent_symbols = self._find_parent_symbols(node)

            chunks.append(
                CodeChunk(
                    content=chunk_content,
                    start_line=start_line,
                    end_line=end_line,
                    symbols=symbols,
                    chunk_type=chunk_type,
                    parent_symbols=parent_symbols,
                )
            )

        # If no chunks were found, create a single chunk for the whole document
        if not chunks:
            chunks.append(
                CodeChunk(
                    content=content,
                    start_line=1,
                    end_line=root_node.end_point[0] + 1,
                    symbols=[],
                    chunk_type="module",
                    parent_symbols=[],
                )
            )

        return chunks

    def _collect_nodes_by_type(
        self, node: tree_sitter.Node, types: List[str], result: List[tree_sitter.Node]
    ) -> None:
        """Collect nodes of specified types from a tree.

        Args:
            node: The current node
            types: Types of nodes to collect
            result: List to store collected nodes
        """
        if node.type in types:
            result.append(node)

        for child in node.children:
            self._collect_nodes_by_type(child, types, result)

    def _find_name_node(self, node: tree_sitter.Node) -> Optional[tree_sitter.Node]:
        """Find the name node of a definition or declaration.

        Args:
            node: The node to find the name for

        Returns:
            The name node if found, None otherwise
        """
        # Different node types have different patterns for where the name is
        if node.type == "function_definition":  # Python
            for child in node.children:
                if child.type == "identifier":
                    return child

        elif node.type == "class_definition":  # Python
            for child in node.children:
                if child.type == "identifier":
                    return child

        elif node.type in ["function_declaration", "class_declaration"]:  # JS/TS
            for child in node.children:
                if child.type == "identifier":
                    return child

        elif node.type == "method_definition":  # JS/TS
            for child in node.children:
                if child.type == "property_identifier":
                    return child

        return None

    def _find_parent_symbols(self, node: tree_sitter.Node) -> List[str]:
        """Find parent symbols of a node.

        Args:
            node: The node to find parents for

        Returns:
            List of parent symbol names
        """
        parents: List[str] = []
        current = node.parent

        while current:
            if current.type in ["class_definition", "class_declaration"]:
                name_node = self._find_name_node(current)
                if name_node:
                    parents.append(self.safe_decode(name_node.text))

            current = current.parent

        return parents

    def _build_symbol_table(
        self, node: tree_sitter.Node, types: List[str]
    ) -> Dict[str, List[int]]:
        """Build a table mapping symbols to line numbers.

        Args:
            node: The root node
            types: Types of nodes to include

        Returns:
            Dictionary mapping symbol names to line numbers
        """
        symbols: Dict[str, List[int]] = {}
        nodes: List[tree_sitter.Node] = []

        self._collect_nodes_by_type(node, types, nodes)

        for node in nodes:
            name_node = self._find_name_node(node)
            if name_node:
                symbol_name = self.safe_decode(name_node.text)
                line_number = node.start_point[0] + 1

                if symbol_name in symbols:
                    symbols[symbol_name].append(line_number)
                else:
                    symbols[symbol_name] = [line_number]

        return symbols

    # Language-specific extractors

    def _extract_python_imports(self, node: tree_sitter.Node) -> List[str]:
        """Extract Python imports from a parsed tree.

        Args:
            node: The root node

        Returns:
            List of import statements
        """
        imports: List[str] = []
        import_nodes: List[tree_sitter.Node] = []

        self._collect_nodes_by_type(
            node, ["import_statement", "import_from_statement"], import_nodes
        )

        for import_node in import_nodes:
            imports.append(self.safe_decode(import_node.text))

        return imports

    def _extract_python_classes(self, node: tree_sitter.Node) -> List[str]:
        """Extract Python class names from a parsed tree.

        Args:
            node: The root node

        Returns:
            List of class names
        """
        classes: List[str] = []
        class_nodes: List[tree_sitter.Node] = []

        self._collect_nodes_by_type(node, ["class_definition"], class_nodes)

        for class_node in class_nodes:
            name_node = self._find_name_node(class_node)
            if name_node:
                classes.append(self.safe_decode(name_node.text))

        return classes

    def _extract_python_functions(self, node: tree_sitter.Node) -> List[str]:
        """Extract Python function names from a parsed tree.

        Args:
            node: The root node

        Returns:
            List of function names
        """
        functions: List[str] = []
        function_nodes: List[tree_sitter.Node] = []

        self._collect_nodes_by_type(node, ["function_definition"], function_nodes)

        for function_node in function_nodes:
            name_node = self._find_name_node(function_node)
            if name_node:
                functions.append(self.safe_decode(name_node.text))

        return functions

    def _extract_js_imports(self, node: tree_sitter.Node) -> List[str]:
        """Extract JavaScript/TypeScript imports from a parsed tree.

        Args:
            node: The root node

        Returns:
            List of import statements
        """
        imports: List[str] = []
        import_nodes: List[tree_sitter.Node] = []

        self._collect_nodes_by_type(node, ["import_statement"], import_nodes)

        for import_node in import_nodes:
            imports.append(self.safe_decode(import_node.text))

        return imports

    def _extract_js_classes(self, node: tree_sitter.Node) -> List[str]:
        """Extract JavaScript/TypeScript class names from a parsed tree.

        Args:
            node: The root node

        Returns:
            List of class names
        """
        classes: List[str] = []
        class_nodes: List[tree_sitter.Node] = []

        self._collect_nodes_by_type(node, ["class_declaration"], class_nodes)

        for class_node in class_nodes:
            name_node = self._find_name_node(class_node)
            if name_node:
                classes.append(self.safe_decode(name_node.text))

        return classes

    def _extract_js_functions(self, node: tree_sitter.Node) -> List[str]:
        """Extract JavaScript/TypeScript function names from a parsed tree.

        Args:
            node: The root node

        Returns:
            List of function names
        """
        functions: List[str] = []
        function_nodes: List[tree_sitter.Node] = []

        self._collect_nodes_by_type(
            node, ["function_declaration", "method_definition"], function_nodes
        )

        for function_node in function_nodes:
            name_node = self._find_name_node(function_node)
            if name_node:
                functions.append(self.safe_decode(name_node.text))

        return functions

    def safe_decode(self, b: Optional[bytes]) -> str:
        return b.decode("utf-8") if b is not None else ""
