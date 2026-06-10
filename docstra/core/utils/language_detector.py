"""
Intelligent language and framework detection for determining appropriate ignore patterns.
"""

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json


class LanguageDetector:
    """Detects programming languages and frameworks in a codebase."""

    # Language detection based on file extensions
    LANGUAGE_EXTENSIONS = {
        "python": [".py", ".pyx", ".pyi"],
        "javascript": [".js", ".mjs", ".cjs"],
        "typescript": [".ts", ".tsx"],
        "java": [".java"],
        "kotlin": [".kt", ".kts"],
        "scala": [".scala"],
        "go": [".go"],
        "rust": [".rs"],
        "c": [".c", ".h"],
        "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".hxx"],
        "csharp": [".cs"],
        "fsharp": [".fs", ".fsx"],
        "vb": [".vb"],
        "php": [".php"],
        "ruby": [".rb"],
        "swift": [".swift"],
        "dart": [".dart"],
        "r": [".r", ".R"],
        "julia": [".jl"],
        "perl": [".pl", ".pm"],
        "lua": [".lua"],
        "shell": [".sh", ".bash", ".zsh", ".fish"],
        "powershell": [".ps1"],
        "elixir": [".ex", ".exs"],
        "erlang": [".erl"],
        "haskell": [".hs"],
        "clojure": [".clj", ".cljs"],
        "elm": [".elm"],
        "ocaml": [".ml", ".mli"],
        "nim": [".nim"],
        "zig": [".zig"],
        "d": [".d"],
        "groovy": [".groovy"],
        "matlab": [".m"],
    }

    # Framework/tool detection based on specific files
    FRAMEWORK_INDICATORS = {
        # Python frameworks
        "django": ["manage.py", "settings.py", "wsgi.py"],
        "flask": ["app.py", "application.py"],
        "fastapi": ["main.py"],
        "poetry": ["pyproject.toml"],
        "uv": ["uv.lock", ".python-version"],
        "pipenv": ["Pipfile"],
        "conda": ["environment.yml", "environment.yaml"],
        # JavaScript/Node.js frameworks
        "nodejs": ["package.json"],
        "npm": ["package-lock.json"],
        "yarn": ["yarn.lock"],
        "pnpm": ["pnpm-lock.yaml"],
        "react": ["package.json"],  # Will check contents
        "vue": ["vue.config.js", "nuxt.config.js"],
        "angular": ["angular.json"],
        "svelte": ["svelte.config.js"],
        "next": ["next.config.js"],
        "gatsby": ["gatsby-config.js"],
        "webpack": ["webpack.config.js"],
        "vite": ["vite.config.js", "vite.config.ts"],
        "rollup": ["rollup.config.js"],
        "parcel": [".parcelrc"],
        # Java frameworks
        "maven": ["pom.xml"],
        "gradle": ["build.gradle", "build.gradle.kts", "gradlew"],
        "spring": ["application.properties", "application.yml"],
        "android": ["AndroidManifest.xml"],
        # .NET frameworks
        "dotnet": ["*.csproj", "*.fsproj", "*.vbproj", "*.sln"],
        "nuget": ["packages.config"],
        # Go
        "go_modules": ["go.mod", "go.sum"],
        # Rust
        "cargo": ["Cargo.toml", "Cargo.lock"],
        # Ruby
        "bundler": ["Gemfile", "Gemfile.lock"],
        "rails": ["config/application.rb"],
        # PHP
        "composer": ["composer.json", "composer.lock"],
        "laravel": ["artisan"],
        # Swift
        "swift_package": ["Package.swift"],
        "xcode": ["*.xcodeproj", "*.xcworkspace"],
        # Dart/Flutter
        "flutter": ["pubspec.yaml"],
        # R
        "r_package": ["DESCRIPTION", "NAMESPACE"],
        # Docker
        "docker": ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"],
        # CI/CD
        "github_actions": [".github/workflows"],
        "gitlab_ci": [".gitlab-ci.yml"],
        "jenkins": ["Jenkinsfile"],
        "travis": [".travis.yml"],
        "circleci": [".circleci/config.yml"],
        # Documentation
        "sphinx": ["conf.py", "docs/conf.py"],
        "mkdocs": ["mkdocs.yml"],
        "gitbook": ["book.json", ".gitbook.yaml"],
        "jekyll": ["_config.yml"],
        # Testing
        "pytest": ["pytest.ini", "pyproject.toml"],
        "jest": ["jest.config.js"],
        "mocha": [".mocharc.json"],
        "karma": ["karma.conf.js"],
        # Linting/Formatting
        "eslint": [".eslintrc.js", ".eslintrc.json", "eslint.config.js"],
        "prettier": [".prettierrc", "prettier.config.js"],
        "black": ["pyproject.toml"],
        "flake8": [".flake8", "setup.cfg"],
        "mypy": ["mypy.ini", "pyproject.toml"],
        "ruff": ["ruff.toml", "pyproject.toml"],
    }

    # Language-specific ignore patterns
    LANGUAGE_IGNORE_PATTERNS = {
        "python": [
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".Python",
            "build/",
            "develop-eggs/",
            "dist/",
            "downloads/",
            "eggs/",
            ".eggs/",
            "lib/",
            "lib64/",
            "parts/",
            "sdist/",
            "var/",
            "wheels/",
            "*.egg-info/",
            ".installed.cfg",
            "*.egg",
            ".pytest_cache/",
            ".coverage",
            "htmlcov/",
            ".tox/",
            ".nox/",
            ".mypy_cache/",
            ".ruff_cache/",
            "venv/",
            ".venv/",
            "env/",
            ".env/",
            "ENV/",
            "env.bak/",
            "venv.bak/",
        ],
        "javascript": [
            "node_modules/",
            "npm-debug.log*",
            "yarn-debug.log*",
            "yarn-error.log*",
            ".npm",
            ".eslintcache",
            ".nyc_output",
            "coverage/",
            "*.tgz",
            "*.tar.gz",
            ".cache/",
            ".parcel-cache/",
            ".next/",
            "out/",
            "build/",
            "dist/",
            ".nuxt/",
            ".vuepress/dist",
            ".serverless/",
            ".fusebox/",
            ".dynamodb/",
            ".tern-port",
        ],
        "typescript": [
            "node_modules/",
            "*.tsbuildinfo",
            ".eslintcache",
            "coverage/",
            "build/",
            "dist/",
            "lib/",
            "*.d.ts.map",
        ],
        "java": [
            "*.class",
            "*.log",
            "*.ctxt",
            ".mtj.tmp/",
            "*.jar",
            "*.war",
            "*.nar",
            "*.ear",
            "*.zip",
            "*.tar.gz",
            "*.rar",
            "hs_err_pid*",
            "target/",
            "build/",
            ".gradle/",
            "gradle-app.setting",
            "!gradle-wrapper.jar",
            ".gradletasknamecache",
            "bin/",
            ".settings/",
            ".metadata",
            ".classpath",
            ".project",
        ],
        "csharp": [
            "bin/",
            "obj/",
            "*.user",
            "*.suo",
            "*.userosscache",
            "*.sln.docstates",
            "[Dd]ebug/",
            "[Dd]ebugPublic/",
            "[Rr]elease/",
            "[Rr]eleases/",
            "x64/",
            "x86/",
            "build/",
            "bld/",
            "[Bb]in/",
            "[Oo]bj/",
            "[Ll]og/",
            ".vs/",
            "packages/",
            "*.nupkg",
            "*.snupkg",
        ],
        "go": [
            "*.exe",
            "*.exe~",
            "*.dll",
            "*.so",
            "*.dylib",
            "*.test",
            "*.out",
            "go.work",
            "vendor/",
        ],
        "rust": [
            "target/",
            "Cargo.lock",
            "**/*.rs.bk",
            "*.pdb",
        ],
        "ruby": [
            "*.gem",
            "*.rbc",
            "/.config",
            "/coverage/",
            "/InstalledFiles",
            "/pkg/",
            "/spec/reports/",
            "/spec/examples.txt",
            "/test/tmp/",
            "/test/version_tmp/",
            "/tmp/",
            ".bundle/",
            "vendor/bundle",
            "lib/bundler/man",
            ".rvmrc",
            ".rbenv-version",
            ".ruby-version",
            ".ruby-gemset",
            ".sass-cache/",
        ],
        "php": [
            "vendor/",
            "composer.phar",
            "composer.lock",
            ".env",
            ".env.backup",
            ".phpunit.result.cache",
            "Homestead.json",
            "Homestead.yaml",
            "npm-debug.log",
            "yarn-error.log",
            "storage/*.key",
            ".env",
            "Homestead.json",
            "Homestead.yaml",
            "auth.json",
        ],
        "swift": [
            "*.xcodeproj/",
            "*.xcworkspace/",
            "xcuserdata/",
            "*.moved-aside",
            "*.pbxuser",
            "!default.pbxuser",
            "*.mode1v3",
            "!default.mode1v3",
            "*.mode2v3",
            "!default.mode2v3",
            "*.perspectivev3",
            "!default.perspectivev3",
            "xcuserdata/",
            "*.xccheckout",
            "*.xcscmblueprint",
            "DerivedData/",
            "*.hmap",
            "*.ipa",
            "*.dSYM.zip",
            "*.dSYM",
            "timeline.xctimeline",
            "playground.xcworkspace",
            ".build/",
            "Packages/",
            "Package.pins",
            "Package.resolved",
        ],
    }

    # Framework-specific ignore patterns
    FRAMEWORK_IGNORE_PATTERNS = {
        "django": [
            "*.log",
            "local_settings.py",
            "db.sqlite3",
            "db.sqlite3-journal",
            "media/",
            "staticfiles/",
            ".env",
        ],
        "react": [
            "build/",
            ".env.local",
            ".env.development.local",
            ".env.test.local",
            ".env.production.local",
        ],
        "vue": [
            ".nuxt/",
            "dist/",
            ".vuepress/dist",
        ],
        "angular": [
            "dist/",
            "tmp/",
            "out-tsc/",
            "bazel-out",
            ".angular/cache",
        ],
        "next": [
            ".next/",
            "out/",
            ".env*.local",
        ],
        "gatsby": [
            ".cache/",
            "public/",
        ],
        "docker": [
            ".dockerignore",
        ],
        "flutter": [
            ".dart_tool/",
            ".flutter-plugins",
            ".flutter-plugins-dependencies",
            ".packages",
            ".pub-cache/",
            ".pub/",
            "build/",
            "ios/Flutter/Flutter.framework",
            "ios/Flutter/Flutter.podspec",
            "ios/Runner/GeneratedPluginRegistrant.*",
        ],
        "android": [
            "*.iml",
            ".gradle",
            "local.properties",
            ".idea/",
            ".DS_Store",
            "build/",
            "captures/",
            ".externalNativeBuild",
            ".cxx",
        ],
        "xcode": [
            "*.xcodeproj/",
            "*.xcworkspace/",
            "xcuserdata/",
            "DerivedData/",
        ],
        "sphinx": [
            "_build/",
            "_static/",
            "_templates/",
        ],
        "mkdocs": [
            "site/",
        ],
    }

    def __init__(self, base_path: str):
        """Initialize the language detector.

        Args:
            base_path: Base path of the codebase to analyze
        """
        self.base_path = Path(base_path).resolve()

    def detect_languages_and_frameworks(self) -> Tuple[Dict[str, int], List[str]]:
        """Detect languages and frameworks in the codebase.

        Returns:
            Tuple of (language_counts, detected_frameworks)
        """
        language_counts: Counter[str] = Counter()
        detected_frameworks = []

        # Scan for files and detect languages
        for root, dirs, files in os.walk(self.base_path):
            # Skip common ignore directories for faster scanning
            dirs[:] = [d for d in dirs if not self._should_skip_dir(d)]

            for file in files:
                file_path = Path(root) / file

                # Detect language by extension
                language = self._detect_language_by_extension(file)
                if language:
                    language_counts[language] += 1

                # Check for framework indicators
                frameworks = self._detect_frameworks_by_file(file_path)
                detected_frameworks.extend(frameworks)

        # Remove duplicates from frameworks
        detected_frameworks = list(set(detected_frameworks))

        # Additional framework detection based on package contents
        additional_frameworks = self._detect_frameworks_by_content()
        detected_frameworks.extend(additional_frameworks)
        detected_frameworks = list(set(detected_frameworks))

        return dict(language_counts), detected_frameworks

    def generate_ignore_patterns(self) -> List[str]:
        """Generate appropriate ignore patterns based on detected languages and frameworks.

        Returns:
            List of ignore patterns
        """
        language_counts, frameworks = self.detect_languages_and_frameworks()

        # Start with universal patterns
        patterns = [
            ".git/",
            ".DS_Store",
            "Thumbs.db",
            "*.log",
            "*.tmp",
            "*.temp",
            ".env",
            ".env.local",
            ".env.*.local",
        ]

        # Only add patterns for languages that have significant presence (>= 5% of files or >= 3 files)
        total_files = sum(language_counts.values())
        significant_threshold = max(3, total_files * 0.05)

        for language, count in language_counts.items():
            if (
                count >= significant_threshold
                and language in self.LANGUAGE_IGNORE_PATTERNS
            ):
                patterns.extend(self.LANGUAGE_IGNORE_PATTERNS[language])

        # Add framework-specific patterns
        for framework in frameworks:
            if framework in self.FRAMEWORK_IGNORE_PATTERNS:
                patterns.extend(self.FRAMEWORK_IGNORE_PATTERNS[framework])

        # Remove duplicates while preserving order
        seen = set()
        unique_patterns = []
        for pattern in patterns:
            if pattern not in seen:
                seen.add(pattern)
                unique_patterns.append(pattern)

        return unique_patterns

    def _should_skip_dir(self, dirname: str) -> bool:
        """Check if a directory should be skipped during scanning.

        Args:
            dirname: Directory name

        Returns:
            True if directory should be skipped
        """
        skip_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            ".mypy_cache",
            ".ruff_cache",
            ".pytest_cache",
            "venv",
            ".venv",
            "env",
            ".env",
            "build",
            "dist",
            "target",
            ".gradle",
            ".vs",
            ".vscode",
            ".idea",
            "bin",
            "obj",
        }
        return dirname in skip_dirs

    def _detect_language_by_extension(self, filename: str) -> str:
        """Detect programming language by file extension.

        Args:
            filename: Name of the file

        Returns:
            Detected language or empty string
        """
        ext = Path(filename).suffix.lower()
        for language, extensions in self.LANGUAGE_EXTENSIONS.items():
            if ext in extensions:
                return language
        return ""

    def _detect_frameworks_by_file(self, file_path: Path) -> List[str]:
        """Detect frameworks by specific files.

        Args:
            file_path: Path to the file

        Returns:
            List of detected frameworks
        """
        frameworks = []
        filename = file_path.name

        for framework, indicators in self.FRAMEWORK_INDICATORS.items():
            for indicator in indicators:
                if "*" in indicator:
                    # Handle glob patterns
                    import fnmatch

                    if fnmatch.fnmatch(filename, indicator):
                        frameworks.append(framework)
                elif indicator.endswith("/"):
                    # Directory indicator
                    if (file_path.parent / indicator.rstrip("/")).is_dir():
                        frameworks.append(framework)
                else:
                    # Exact file match
                    if filename == indicator or file_path.name == indicator:
                        frameworks.append(framework)

        return frameworks

    def _detect_frameworks_by_content(self) -> List[str]:
        """Detect frameworks by analyzing file contents.

        Returns:
            List of detected frameworks
        """
        frameworks = []

        # Check package.json for JavaScript frameworks
        package_json = self.base_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json, "r") as f:
                    data = json.load(f)

                dependencies = {
                    **data.get("dependencies", {}),
                    **data.get("devDependencies", {}),
                }

                if "react" in dependencies or "react-dom" in dependencies:
                    frameworks.append("react")
                if "vue" in dependencies:
                    frameworks.append("vue")
                if "@angular/core" in dependencies:
                    frameworks.append("angular")
                if "svelte" in dependencies:
                    frameworks.append("svelte")
                if "next" in dependencies:
                    frameworks.append("next")
                if "gatsby" in dependencies:
                    frameworks.append("gatsby")
                if "webpack" in dependencies:
                    frameworks.append("webpack")
                if "vite" in dependencies:
                    frameworks.append("vite")
                if "jest" in dependencies:
                    frameworks.append("jest")
                if "eslint" in dependencies:
                    frameworks.append("eslint")
                if "prettier" in dependencies:
                    frameworks.append("prettier")

            except (json.JSONDecodeError, IOError):
                pass

        # Check pyproject.toml for Python tools
        pyproject_toml = self.base_path / "pyproject.toml"
        if pyproject_toml.exists():
            try:
                content = pyproject_toml.read_text()
                if "[tool.poetry]" in content:
                    frameworks.append("poetry")
                if "[tool.uv]" in content or 'build-backend = "uv_build"' in content:
                    frameworks.append("uv")
                if "[tool.black]" in content:
                    frameworks.append("black")
                if "[tool.mypy]" in content:
                    frameworks.append("mypy")
                if "[tool.ruff]" in content:
                    frameworks.append("ruff")
                if "[tool.pytest]" in content:
                    frameworks.append("pytest")
            except IOError:
                pass

        return frameworks

    def get_detection_summary(self) -> Dict[str, Any]:
        """Get a summary of the detection results.

        Returns:
            Dictionary containing detection summary
        """
        language_counts, frameworks = self.detect_languages_and_frameworks()
        patterns = self.generate_ignore_patterns()

        # Calculate primary language
        primary_language = (
            max(language_counts.items(), key=lambda x: x[1])[0]
            if language_counts
            else "unknown"
        )

        return {
            "primary_language": primary_language,
            "languages": language_counts,
            "frameworks": frameworks,
            "ignore_patterns": patterns,
            "total_patterns": len(patterns),
            "codebase_type": self._classify_codebase_type(language_counts, frameworks),
        }

    def _classify_codebase_type(
        self, language_counts: Dict[str, int], frameworks: List[str]
    ) -> str:
        """Classify the type of codebase.

        Args:
            language_counts: Dictionary of language counts
            frameworks: List of detected frameworks

        Returns:
            Codebase type classification
        """
        if not language_counts:
            return "unknown"

        primary_language = max(language_counts.items(), key=lambda x: x[1])[0]

        # Web development
        if any(
            fw in frameworks
            for fw in ["react", "vue", "angular", "svelte", "next", "gatsby"]
        ):
            return "web_frontend"
        elif "nodejs" in frameworks or primary_language in ["javascript", "typescript"]:
            return "web_backend"

        # Mobile development
        elif any(fw in frameworks for fw in ["flutter", "android", "xcode"]):
            return "mobile"

        # Data science/ML
        elif primary_language == "python" and any(
            fw in frameworks for fw in ["jupyter", "conda"]
        ):
            return "data_science"

        # Desktop applications
        elif primary_language in ["csharp", "java", "cpp", "swift"]:
            return "desktop"

        # System programming
        elif primary_language in ["rust", "go", "c"]:
            return "system"

        # Library/package
        elif any(
            fw in frameworks
            for fw in ["poetry", "uv", "cargo", "maven", "gradle", "composer"]
        ):
            return "library"

        # Default to the primary language
        return primary_language
