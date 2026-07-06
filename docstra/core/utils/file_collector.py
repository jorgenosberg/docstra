import re
import logging
from pathlib import Path
from typing import List, Optional, Pattern, Union, Dict, cast

# Directories no one wants indexed or documented: dependency trees, VCS
# internals, caches, and build output. Pruned during the walk so large
# trees like .venv are never traversed. Explicit include_dirs still win,
# and a .docstraignore can add project-specific patterns on top.
DEFAULT_EXCLUDE_DIRS: List[str] = [
    ".git/",
    ".hg/",
    ".svn/",
    ".docstra/",
    "node_modules/",
    "bower_components/",
    ".venv/",
    "venv/",
    "env/",
    ".env/",
    "virtualenv/",
    "site-packages/",
    "__pycache__/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".tox/",
    ".nox/",
    ".eggs/",
    "*.egg-info/",
    "dist/",
    "build/",
    "target/",
    "out/",
    ".gradle/",
    ".idea/",
    ".vscode/",
    ".cache/",
    ".next/",
    ".nuxt/",
    ".terraform/",
    "htmlcov/",
    "coverage/",
    "vendor/",
]


class GitIgnorePattern:
    """A single gitignore pattern with matching functionality."""

    def __init__(self, pattern: str):
        """Initialize a gitignore pattern.

        Args:
            pattern: The gitignore pattern string
        """
        self.original_pattern = pattern
        self.pattern = self._parse_pattern(pattern)

        # Extract pattern properties
        self.is_negated = pattern.startswith("!")
        self.is_dir_only = pattern.endswith("/")
        self.is_absolute = (
            pattern.startswith("/") or "/" in pattern[:-1]
            if pattern.endswith("/")
            else "/" in pattern
        )

        # Convert to regex for faster matching
        self.regex = self._pattern_to_regex(self.pattern)

    def _parse_pattern(self, pattern: str) -> str:
        """Parse the raw gitignore pattern.

        Args:
            pattern: The raw pattern

        Returns:
            Processed pattern
        """
        # Handle negation
        if pattern.startswith("!"):
            pattern = pattern[1:]

        # Remove trailing spaces unless escaped
        if pattern.endswith(" ") and not pattern.endswith("\\ "):
            pattern = pattern.rstrip()

        # Handle escaped characters
        pattern = self._unescape_pattern(pattern)

        return pattern

    def _unescape_pattern(self, pattern: str) -> str:
        """Unescape special characters in the pattern.

        Args:
            pattern: Pattern string

        Returns:
            Unescaped pattern
        """
        # Replace \# with # and \! with ! etc.
        result = ""
        i = 0
        while i < len(pattern):
            if pattern[i] == "\\" and i + 1 < len(pattern):
                result += pattern[i + 1]
                i += 2
            else:
                result += pattern[i]
                i += 1
        return result

    def _pattern_to_regex(self, pattern: str) -> Pattern:
        """Convert a gitignore pattern to a regex pattern.

        Args:
            pattern: The gitignore pattern

        Returns:
            Compiled regex pattern
        """
        # Handle patterns without slashes - these should match at any level
        if "/" not in pattern:
            # No slash: pattern is a glob for the basename (file or directory name).
            # It should match a file or directory name component anywhere in the tree.
            # Example: "*.log" should match "file.log", "dir/file.log"
            # Example: "foo" should match "foo", "dir/foo"

            # Convert simple gitignore glob to regex for a path component:
            # '*' -> '[^/]*' (matches anything except slash within a component)
            # '?' -> '[^/]'  (matches any single char except slash within a component)
            # '.' -> '\.' (literal dot)
            # other chars -> literal (escaped)
            component_glob_regex = ""
            i = 0
            while i < len(pattern):
                char = pattern[i]
                if char == "*":
                    component_glob_regex += "[^/]*"
                elif char == "?":
                    component_glob_regex += "[^/]"
                elif char == "[":  # Character class [abc]
                    j = i
                    while j < len(pattern) and pattern[j] != "]":
                        j += 1
                    if j < len(pattern) and pattern[j] == "]":
                        component_glob_regex += re.escape(pattern[i : j + 1])
                        i = j
                    else:  # Unclosed bracket, treat as literal
                        component_glob_regex += re.escape(char)
                else:
                    component_glob_regex += re.escape(char)
                i += 1

            # Regex to match this glob as a full path component (file or directory name)
            # anchored by slashes or start/end of path.
            # It should match 'component' or 'path/component' or 'component/' or 'path/component/'
            # So, we construct regex: (^|/)component_regex_pattern(/|$) to match a full component.
            # If self.is_dir_only is true (e.g. pattern was "foo*/" but no actual '/' char in `pattern` after parsing),
            # we ensure it matches as a directory.
            # However, self.is_dir_only is based on original pattern ending with /
            # if pattern had no actual '/' (e.g. just "foo*"), self.is_dir_only will be False.
            if self.is_dir_only:  # e.g. original pattern was "temp*/"
                regex = f"(^|/)({component_glob_regex})/$"  # Must end with slash
            else:  # e.g. original pattern was "*.log" or "temp*"
                # This will match "component_glob_regex" as a filename or directory name.
                # Example: For "*.log", matches "file.log" in "path/file.log" or "file.log/" in "path/file.log/"
                regex = f"(^|/)({component_glob_regex})(/|$)"
            return re.compile(regex)

        # For patterns with slashes, standard processing follows:

        # Start at beginning of string or after a slash
        if pattern.startswith("/"):
            regex = "^"
            pattern = pattern[1:]
        else:
            regex = "(^|/)"

        # Remove trailing slash if present
        if pattern.endswith("/"):
            pattern = pattern[:-1]
            dir_only = True
        else:
            dir_only = False

        # Process the pattern
        i = 0
        while i < len(pattern):
            if pattern[i] == "*":
                if i + 1 < len(pattern) and pattern[i + 1] == "*":
                    # Double star - match any number of directories
                    regex += ".*"
                    i += 2
                else:
                    # Single star - match any characters except slash
                    regex += "[^/]*"
                    i += 1
            elif pattern[i] == "?":
                # Question mark - match any single character except slash
                regex += "[^/]"
                i += 1
            elif pattern[i] == "[":
                # Character class
                j = i + 1
                while j < len(pattern) and pattern[j] != "]":
                    j += 1
                if j < len(pattern):
                    regex += pattern[i : j + 1]
                    i = j + 1
                else:
                    # Unclosed character class - treat as literal
                    regex += re.escape(pattern[i])
                    i += 1
            else:
                # Regular character
                regex += re.escape(pattern[i])
                i += 1

        # Add trailing slash for directory-only patterns
        if dir_only:
            regex += "/"

        # End of string
        regex += "$"

        return re.compile(regex)

    def matches(self, path: str, is_dir: bool = False) -> bool:
        """Check if the pattern matches a path.

        Args:
            path: Path to check
            is_dir: Whether the path is a directory

        Returns:
            True if the pattern matches, False otherwise
        """
        # Directory-only patterns only match directories
        if self.is_dir_only and not is_dir:
            return False

        # The regex is already compiled to handle specifics of gitignore pattern matching
        # (e.g., anchoring, matching anywhere, wildcards).
        # So, we just need to search the given path with the regex.
        return bool(self.regex.search(path))


class GitIgnoreMatcher:
    """Matcher for multiple gitignore patterns."""

    def __init__(self, patterns: List[str]):
        """Initialize with a list of patterns.

        Args:
            patterns: List of gitignore pattern strings
        """
        self.patterns = []

        # Parse each pattern
        for pattern in patterns:
            # Skip blank lines and comments
            pattern = pattern.strip()
            if not pattern or pattern.startswith("#"):
                continue

            self.patterns.append(GitIgnorePattern(pattern))

    def matches(self, path: str, is_dir: bool = False) -> bool:
        """Check if any pattern matches the path.

        Args:
            path: Path to check
            is_dir: Whether the path is a directory

        Returns:
            True if the path should be excluded, False otherwise
        """
        excluded = False

        # Check each pattern in order
        for pattern in self.patterns:
            if pattern.matches(path, is_dir):
                if pattern.is_negated:
                    # Negated pattern - include the file
                    excluded = False
                else:
                    # Regular pattern - exclude the file
                    excluded = True

        return excluded


class FileCollector:
    """Utility for collecting files with inclusion/exclusion rules using gitignore patterns."""

    def __init__(
        self,
        base_path: Union[str, Path],
        include_dirs: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None,
        exclude_files: Optional[List[str]] = None,
        file_extensions: Optional[List[str]] = None,
        log_level: int = logging.INFO,
        use_default_excludes: bool = True,
    ):
        """Initialize the file collector.

        Args:
            base_path: Base path for file collection
            include_dirs: List of directories to specifically include
            exclude_dirs: List of gitignore-style patterns for directories to exclude
            exclude_files: List of gitignore-style patterns for files to exclude
            file_extensions: List of file extensions to include
            log_level: Logging level
            use_default_excludes: Prune DEFAULT_EXCLUDE_DIRS (dependency trees,
                caches, build output) during the walk
        """
        self.base_path = Path(base_path).resolve()
        self.include_dirs = include_dirs or []
        self.file_extensions = file_extensions or []
        if use_default_excludes:
            exclude_dirs = DEFAULT_EXCLUDE_DIRS + (exclude_dirs or [])

        # Set up logging
        self.logger = logging.getLogger("docstra.file_collector")
        self.logger.setLevel(log_level)

        # Add console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter("%(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Create matchers for directory and file exclusions
        self.dir_matcher = GitIgnoreMatcher(exclude_dirs or [])
        self.file_matcher = GitIgnoreMatcher(exclude_files or [])

        # Statistics with explicit types
        self.stats: Dict[str, Union[int, Dict[str, int]]] = {
            "visited_dirs": 0,
            "visited_files": 0,
            "included_files": 0,
            "excluded_dirs": 0,
            "excluded_files": 0,
            "dir_counts": {},  # Directory -> count of included files
        }

    @staticmethod
    def default_code_file_extensions() -> List[str]:
        """Get the default list of code file extensions.

        Returns:
            List of common code file extensions
        """
        return [
            # Common languages
            ".py",  # Python
            ".js",  # JavaScript
            ".ts",  # TypeScript
            ".java",  # Java
            ".go",  # Go
            ".rs",  # Rust
            ".c",  # C
            ".cpp",  # C++
            ".cc",  # C++ alternative
            ".h",  # C/C++ header
            ".hpp",  # C++ header
            ".cs",  # C#
            ".rb",  # Ruby
            ".php",  # PHP
            ".swift",  # Swift
            ".kt",  # Kotlin
            # Additional languages
            ".r",  # R
            ".jl",  # Julia
            ".scala",  # Scala
            ".fs",  # F#
            ".fsx",  # F# script
            ".pl",  # Perl
            ".pm",  # Perl module
            ".sh",  # Shell script
            ".bash",  # Bash script
            ".zsh",  # Zsh script
            ".ps1",  # PowerShell
            ".groovy",  # Groovy
            ".lua",  # Lua
            ".m",  # Objective-C / MATLAB
            ".mm",  # Objective-C++
            ".clj",  # Clojure
            ".erl",  # Erlang
            ".ex",  # Elixir
            ".exs",  # Elixir script
            ".elm",  # Elm
            ".hs",  # Haskell
            ".dart",  # Dart
            ".d",  # D language
            ".vb",  # Visual Basic
            ".sql",  # SQL
            # Web development
            ".jsx",  # React JSX
            ".tsx",  # React TSX
            ".html",  # HTML
            ".htm",  # HTML alternative
            ".css",  # CSS
            ".scss",  # SCSS
            ".sass",  # Sass
            ".less",  # Less
            ".vue",  # Vue
            ".svelte",  # Svelte
            # Configuration and data formats
            ".json",  # JSON
            ".yaml",  # YAML
            ".yml",  # YAML alternative
            ".xml",  # XML
            ".toml",  # TOML
            ".ini",  # INI configuration
            ".proto",  # Protocol Buffers
            # Documentation
            ".md",  # Markdown
            ".rst",  # reStructuredText
        ]

    def collect_files(self) -> List[Path]:
        """Collect files according to inclusion/exclusion rules.

        Returns:
            List of collected file paths
        """
        self.logger.info(f"Starting file collection from {self.base_path}")
        self.logger.debug(f"Include dirs: {self.include_dirs}")
        self.logger.debug(f"File extensions: {self.file_extensions}")

        # Reset statistics with explicit types
        self.stats = {
            "visited_dirs": 0,
            "visited_files": 0,
            "included_files": 0,
            "excluded_dirs": 0,
            "excluded_files": 0,
            "dir_counts": {},
        }

        collected_files: List[Path] = []

        # Handle single file case
        if self.base_path.is_file():
            self.stats["visited_files"] = cast(int, self.stats["visited_files"]) + 1
            if self._should_include_file(self.base_path):
                collected_files.append(self.base_path)
                self.stats["included_files"] = (
                    cast(int, self.stats["included_files"]) + 1
                )
            else:
                self.stats["excluded_files"] = (
                    cast(int, self.stats["excluded_files"]) + 1
                )

            self._log_statistics()
            return collected_files

        # Recursively walk the directory tree
        for file_path in self._walk_directory(self.base_path):
            collected_files.append(file_path)

        # Log statistics and potential issues
        self._log_statistics()

        return collected_files

    def _walk_directory(self, dir_path: Path) -> List[Path]:
        """Walk a directory recursively, filtering files according to rules.

        Args:
            dir_path: Directory path to walk

        Returns:
            List of included file paths
        """
        self.stats["visited_dirs"] = cast(int, self.stats["visited_dirs"]) + 1
        included_files: List[Path] = []

        try:
            # Get all directories and files in the current directory
            dirs: List[Path] = []
            files: List[Path] = []

            for path in dir_path.iterdir():
                if path.is_dir():
                    dirs.append(path)
                elif path.is_file():
                    files.append(path)

            # Process each file in the current directory
            for file_path in files:
                self.stats["visited_files"] = cast(int, self.stats["visited_files"]) + 1
                rel_file = str(file_path.relative_to(self.base_path))

                if self._should_include_file(file_path, rel_file):
                    included_files.append(file_path)
                    self.stats["included_files"] = (
                        cast(int, self.stats["included_files"]) + 1
                    )

                    # Update directory count
                    rel_dir = str(file_path.parent.relative_to(self.base_path)) or "."
                    dir_counts = cast(Dict[str, int], self.stats["dir_counts"])
                    dir_counts[rel_dir] = dir_counts.get(rel_dir, 0) + 1
                else:
                    self.stats["excluded_files"] = (
                        cast(int, self.stats["excluded_files"]) + 1
                    )

            # Recursively process subdirectories
            for subdir in dirs:
                rel_dir = str(subdir.relative_to(self.base_path))

                if not self._should_exclude_directory(rel_dir):
                    included_files.extend(self._walk_directory(subdir))
                else:
                    self.logger.debug(f"Excluding directory: {rel_dir}")
                    self.stats["excluded_dirs"] = (
                        cast(int, self.stats["excluded_dirs"]) + 1
                    )

        except (PermissionError, OSError) as e:
            self.logger.warning(f"Error accessing {dir_path}: {e}")

        return included_files

    def _should_exclude_directory(self, rel_dir: str) -> bool:
        """Check if a directory should be excluded.

        Args:
            rel_dir: Relative directory path from base_path

        Returns:
            True if the directory should be excluded, False otherwise
        """
        # Always include specified directories if provided in include_dirs
        if self.include_dirs:
            for include_dir in self.include_dirs:
                include_path = Path(include_dir)
                rel_path = Path(rel_dir)

                # Check if this directory is included or is a subdirectory of an included directory
                if rel_path == include_path or any(
                    parent == include_path for parent in rel_path.parents
                ):
                    return False  # Do not exclude if explicitly included or child of include

        # Check if the full relative directory path matches an exclusion pattern
        # from self.dir_matcher.
        # Directory patterns in dir_matcher (e.g., from "node_modules/")
        # often result in regexes like `(^|/)node_modules/$`.
        # The `rel_dir` path (e.g., "node_modules" or "path/node_modules")
        # typically doesn't have a trailing slash.
        # To ensure correct matching, we form a path-like string for the check
        # that includes a trailing slash if `rel_dir` represents a directory name.
        path_to_check = rel_dir
        if path_to_check != "." and not path_to_check.endswith("/"):
            path_to_check += "/"

        # If self.dir_matcher has a pattern "foo/", its regex is `(^|/)foo/$`.
        # If rel_dir is "foo", path_to_check is "foo/". `dir_matcher.matches("foo/", is_dir=True)` should work.
        # If rel_dir is "bar/foo", path_to_check is "bar/foo/". `dir_matcher.matches("bar/foo/", is_dir=True)` should work.
        if self.dir_matcher.matches(path_to_check, is_dir=True):
            self.logger.debug(
                f"Excluding directory '{rel_dir}' (checked as '{path_to_check}') due to dir_matcher pattern."
            )
            return True

        # If include_dirs is specified, and we haven't returned False above (meaning it wasn't explicitly included),
        # then we should exclude this directory if it doesn't fall under any include_dir path.
        if self.include_dirs:
            # This logic now acts as an explicit include-only mode if include_dirs is non-empty
            is_explicitly_included = False
            for include_dir in self.include_dirs:
                include_path = Path(include_dir)
                rel_path = Path(rel_dir)
                if rel_path == include_path or any(
                    parent == include_path for parent in rel_path.parents
                ):
                    is_explicitly_included = True
                    break
            if not is_explicitly_included:
                self.logger.debug(
                    f"Excluding directory '{rel_dir}' because it is not in specified include_dirs."
                )
                return True

        return False

    def _should_include_file(
        self, file_path: Path, rel_file: Optional[str] = None
    ) -> bool:
        """Check if a file should be included.

        Args:
            file_path: Path to the file
            rel_file: Optional relative path from base_path

        Returns:
            True if the file should be included, False otherwise
        """
        # Get relative path if not provided
        if rel_file is None:
            rel_file = str(file_path.relative_to(self.base_path))

        # Check file extension
        if self.file_extensions:
            if file_path.suffix.lower() not in self.file_extensions:
                return False

        # Check if the full relative file path matches an exclusion pattern
        if self.file_matcher.matches(rel_file, is_dir=False):
            return False

        # If we've made it here, include the file
        return True

    def _log_statistics(self) -> None:
        """Log collection statistics and potential issues."""
        self.logger.info(
            f"Visited {self.stats['visited_dirs']} directories and {self.stats['visited_files']} files"
        )
        self.logger.info(
            f"Collected {self.stats['included_files']} files, excluded {self.stats['excluded_files']} files and {self.stats['excluded_dirs']} directories"
        )

        # Log top directories with most files
        dir_counts = cast(Dict[str, int], self.stats["dir_counts"])
        if dir_counts:
            self.logger.info("Top directories with most collected files:")
            for dir_path, count in sorted(
                dir_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                self.logger.info(f"  {dir_path}: {count} files")

        # Check for potential issues
        self._check_for_issues()

    def _check_for_issues(self) -> None:
        """Check for potential issues with the file collection."""
        if (
            cast(int, self.stats["included_files"]) < 5
            and cast(int, self.stats["visited_files"]) > 0
        ):
            self.logger.warning(
                "Very few files were included. Check your exclusion patterns or file extensions."
            )

        if cast(int, self.stats["included_files"]) > 5000:
            self.logger.warning(
                f"Unusually large number of files collected ({self.stats['included_files']}). This might impact performance."
            )

        # Check for directories with excessive files
        dir_counts = cast(Dict[str, int], self.stats["dir_counts"])
        for dir_path, count in dir_counts.items():
            if count > 500:
                self.logger.warning(
                    f"Directory '{dir_path}' contains {count} files. Consider excluding this directory if it's not needed."
                )

        # Check for potential patterns that should have been excluded
        problem_patterns = [
            "node_modules",
            "dist",
            "build",
            "__pycache__",
            ".mypy_cache",
            ".ruff_cache",
            ".pytest_cache",
            ".git",
            "venv",
            ".venv",
            "env",
        ]
        for pattern in problem_patterns:
            # Check if any directories with this pattern have files.
            # Match whole path components so '.github' does not match '.git'.
            dir_counts = cast(Dict[str, int], self.stats["dir_counts"])
            matching_dirs = [d for d in dir_counts if pattern in Path(d).parts]
            if matching_dirs:
                total_files = sum(dir_counts[d] for d in matching_dirs)
                self.logger.warning(
                    f"Found {total_files} files in {len(matching_dirs)} '{pattern}' directories. Consider adding exclusion pattern."
                )


def collect_files(
    base_path: Union[str, Path],
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    exclude_files: Optional[List[str]] = None,
    file_extensions: Optional[List[str]] = None,
    log_level: int = logging.INFO,
) -> List[Path]:
    """Collect files with inclusion/exclusion rules."""

    base_path = Path(base_path).resolve()
    dot_docstra_ignore_path = base_path / ".docstra" / ".docstraignore"

    gitignore_patterns = []
    if dot_docstra_ignore_path.exists():
        try:
            with open(dot_docstra_ignore_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        gitignore_patterns.append(line)
            logging.info(
                f"Loaded .docstraignore patterns from {dot_docstra_ignore_path}"
            )
        except Exception as e:
            logging.warning(
                f"Could not read .docstraignore file {dot_docstra_ignore_path}: {e}"
            )

    # Merge universal patterns with command-specific patterns
    # Don't separate them - let GitIgnoreMatcher handle both directory and file patterns
    all_exclude_dirs = (exclude_dirs or []) + gitignore_patterns
    all_exclude_files = (exclude_files or []) + gitignore_patterns

    collector = FileCollector(
        base_path=base_path,
        include_dirs=include_dirs,
        exclude_dirs=all_exclude_dirs,
        exclude_files=all_exclude_files,
        file_extensions=file_extensions,
        log_level=log_level,
    )
    # collect_files now returns files after applying universal ignores and base filters
    universally_included_files = collector.collect_files()

    # The caller is responsible for applying command-specific include/exclude rules
    # based on the config file if needed.

    return universally_included_files


def filter_files_with_patterns(
    file_paths: List[Path],
    base_path: Union[str, Path],
    include_dirs: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    is_dir_check_needed: bool = True,  # Flag to indicate if we need to check if path is a directory
) -> List[Path]:
    """Filters a list of file paths based on include directories and exclude patterns.

    Args:
        file_paths: List of file paths to filter.
        base_path: The base path relative to which include/exclude patterns are applied.
        include_dirs: List of directories to specifically include.
        exclude_patterns: List of gitignore-style patterns to exclude.
        is_dir_check_needed: Whether to check if a path is a directory when applying patterns.

    Returns:
        A list of filtered file paths.
    """
    base_path = Path(base_path).resolve()

    # Create GitIgnoreMatcher for exclude patterns
    exclude_matcher = GitIgnoreMatcher(exclude_patterns or [])

    filtered_files: List[Path] = []
    for file_path in file_paths:
        # Ensure the file path is relative to the base_path for pattern matching
        try:
            rel_path = file_path.relative_to(base_path)
        except ValueError:
            # If file_path is not relative to base_path, include it by default
            # or handle as per requirement. For now, let's assume it should be skipped
            # if not under the base_path.
            continue

        # Apply include directory logic (if any)
        should_include = True
        if include_dirs:
            should_include = False  # Assume exclusion unless explicitly included
            for include_dir in include_dirs:
                include_path = base_path / include_dir
                if file_path == include_path or include_path in file_path.parents:
                    should_include = True
                    break
            if not should_include:
                continue  # Exclude if not in an include directory

        # Apply exclude patterns
        # Check both the file path and potentially its parent directories against exclude patterns
        # gitignore patterns can match directories and prevent walking into them.
        # Here we've already collected files, so we check the file path itself
        # and its parent directories against directory exclude patterns.

        # Check against file exclusion patterns
        if exclude_matcher.matches(str(rel_path), is_dir=False):
            continue  # Exclude if matches file pattern

        # Check parent directories against directory exclusion patterns
        # This is a bit redundant if collect_files already applied universal dir excludes,
        # but necessary for command-specific directory excludes.
        # Need to instantiate a dir_matcher here, or modify FileCollector to expose it.
        # Let's simplify for now and rely on the GitIgnoreMatcher handling both.
        # Re-evaluating: FileCollector uses *separate* matchers for dirs and files.
        # filter_files_with_patterns should probably do the same for consistency with GitIgnore rules.
        # Let's revise this function.
        parts: tuple[str, ...] = (
            Path(rel_path).parts if not isinstance(rel_path, tuple) else rel_path
        )
        is_dir_excluded = False
        current_rel_dir: Path = Path()
        for part in parts[:-1]:  # Check parent directories
            current_rel_dir = current_rel_dir / part
            if exclude_matcher.matches(str(current_rel_dir), is_dir=True):
                is_dir_excluded = True
                break
        if is_dir_excluded:
            continue  # Exclude if in an excluded directory

        # Check against file exclusion patterns
        if exclude_matcher.matches(str(rel_path), is_dir=False):
            continue  # Exclude if matches file pattern

        # If not excluded, include the file
        filtered_files.append(file_path)

    return filtered_files
