"""
Context-aware retrieval system for optimized LLM prompting.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from docstra.core.indexing.code_index import CodebaseIndex
from docstra.core.indexing.repo_map import RepositoryMap
from docstra.core.retrieval.chroma import ChromaRetriever
from docstra.core.retrieval.hybrid import HybridRetriever
from docstra.core.utils.token_counter import ContextBudgetManager, TokenCounter


class QueryIntent(Enum):
    """Types of query intents for targeted retrieval."""
    ARCHITECTURAL = "architectural"
    IMPLEMENTATION = "implementation"
    USAGE = "usage"
    DEBUGGING = "debugging"
    GENERAL = "general"


class QueryAnalysis:
    """Analysis of a user query to determine retrieval strategy."""
    
    def __init__(
        self,
        intent: QueryIntent = QueryIntent.GENERAL,
        symbols: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        technologies: Optional[List[str]] = None,
        complexity: str = "medium"
    ):
        self.intent = intent
        self.symbols = symbols or []
        self.files = files or []
        self.technologies = technologies or []
        self.complexity = complexity  # "low", "medium", "high"
        
    @property
    def is_architectural(self) -> bool:
        """Check if query is about architecture/design."""
        return self.intent == QueryIntent.ARCHITECTURAL
    
    @property 
    def is_implementation(self) -> bool:
        """Check if query is about implementation details."""
        return self.intent == QueryIntent.IMPLEMENTATION
    
    @property
    def is_usage(self) -> bool:
        """Check if query is about how to use something."""
        return self.intent == QueryIntent.USAGE


class ContextAwareRetriever:
    """Adaptive retrieval system that optimizes for model context windows."""
    
    def __init__(
        self,
        base_retriever: ChromaRetriever,
        budget_manager: ContextBudgetManager,
        code_index: Optional[CodebaseIndex] = None,
        repo_map: Optional[RepositoryMap] = None
    ):
        self.base_retriever = base_retriever
        self.budget_manager = budget_manager
        self.code_index = code_index
        self.repo_map = repo_map
        
        # Create hybrid retriever if code index available
        if code_index:
            self.hybrid_retriever = HybridRetriever(base_retriever, code_index)
        else:
            self.hybrid_retriever = None
    
    def retrieve_with_budget(
        self,
        query: str,
        context_type: str = "query",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Retrieve context optimized for available token budget."""
        
        # Analyze query for better targeting
        query_analysis = self._analyze_query(query)
        
        # Get base context budget
        budget = self.budget_manager.get_context_budget()
        
        # Allocate budget based on query intent
        if query_analysis.is_architectural:
            context = self._get_architectural_context(query, query_analysis, budget)
        elif query_analysis.is_implementation:
            context = self._get_implementation_context(query, query_analysis, budget)
        elif query_analysis.is_usage:
            context = self._get_usage_context(query, query_analysis, budget)
        else:
            context = self._get_general_context(query, query_analysis, budget)
        
        return context
    
    def _analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to determine intent and extract key information."""
        
        query_lower = query.lower()
        
        # Intent detection patterns
        architectural_patterns = [
            r'\b(architecture|design|structure|overview|how.*work|organization)\b',
            r'\b(system|module|component|relationship|dependency)\b',
            r'\b(flow|process|pattern|framework)\b',
            r'\b(where.*defined|where.*located|where.*find|location.*of)\b',
            r'\b(which.*file|what.*file|file.*contain)\b',
            r'\b(what.*defined|what.*functions|what.*classes|what.*in.*\.py)\b',
            r'\b(cli\.py|main\.py|app\.py|server\.py)\b'
        ]
        
        implementation_patterns = [
            r'\b(implement|code|algorithm|function|method|class)\b',
            r'\b(how.*implement|how.*code|how.*build)\b',
            r'\b(detail|specific|actual|concrete)\b'
        ]
        
        usage_patterns = [
            r'\b(how.*use|how.*call|example|usage|tutorial)\b',
            r'\b(getting started|quick start|guide)\b',
            r'\b(api|interface|public)\b'
        ]
        
        debugging_patterns = [
            r'\b(error|bug|issue|problem|fix|debug)\b',
            r'\b(why.*not|what.*wrong|troubleshoot)\b'
        ]
        
        # Determine intent
        intent = QueryIntent.GENERAL
        if any(re.search(pattern, query_lower) for pattern in architectural_patterns):
            intent = QueryIntent.ARCHITECTURAL
        elif any(re.search(pattern, query_lower) for pattern in implementation_patterns):
            intent = QueryIntent.IMPLEMENTATION
        elif any(re.search(pattern, query_lower) for pattern in usage_patterns):
            intent = QueryIntent.USAGE
        elif any(re.search(pattern, query_lower) for pattern in debugging_patterns):
            intent = QueryIntent.DEBUGGING
        
        # Extract symbols (functions, classes, variables)
        symbols = self._extract_symbols_from_query(query)
        
        # Extract file references
        files = self._extract_file_references(query)
        
        # Determine complexity based on query length and specificity
        complexity = "low"
        if len(query.split()) > 10 or len(symbols) > 2:
            complexity = "medium"
        if len(query.split()) > 20 or len(symbols) > 4:
            complexity = "high"
        
        return QueryAnalysis(
            intent=intent,
            symbols=symbols,
            files=files,
            complexity=complexity
        )
    
    def _extract_symbols_from_query(self, query: str) -> List[str]:
        """Extract potential code symbols from query."""
        
        # Look for camelCase, snake_case, and PascalCase identifiers
        symbol_patterns = [
            r'\b[a-z][a-zA-Z0-9_]*[A-Z][a-zA-Z0-9_]*\b',  # camelCase
            r'\b[A-Z][a-zA-Z0-9_]*\b',                     # PascalCase
            r'\b[a-z][a-z0-9_]*_[a-z0-9_]*\b',            # snake_case
            r'\b[a-z][a-z0-9_]*\(\)',                     # function calls
        ]
        
        symbols = []
        for pattern in symbol_patterns:
            matches = re.findall(pattern, query)
            symbols.extend(matches)
        
        # Clean up and deduplicate
        symbols = [s.rstrip('()') for s in symbols]
        return list(set(symbols))
    
    def _extract_file_references(self, query: str) -> List[str]:
        """Extract file references from query."""
        
        # Look for file extensions and paths
        file_patterns = [
            r'\b[\w/.-]+\.(py|js|ts|java|cpp|c|h|go|rs|rb|php)\b',
            r'\b[\w.-]+\.[\w]+\b'
        ]
        
        files = []
        for pattern in file_patterns:
            matches = re.findall(pattern, query)
            files.extend(matches)
        
        return list(set(files))
    
    def _get_architectural_context(
        self,
        query: str,
        analysis: QueryAnalysis,
        budget: int
    ) -> Dict[str, Any]:
        """Get architectural context prioritizing repo map and dependencies."""
        
        context_parts = {}
        remaining_budget = budget
        
        # Determine if this is a high-budget context (detailed mode)
        is_detailed_mode = budget > 50000  # More than 50k tokens available
        
        # 1. Repository overview (higher priority and detail for large budgets)
        if self.repo_map and remaining_budget > 500:
            if is_detailed_mode:
                repo_overview = self._get_detailed_repo_overview()
            else:
                repo_overview = self._get_compact_repo_overview()
                
            if repo_overview:
                overview_tokens = self.budget_manager.token_counter.count_tokens(repo_overview)
                # Use more budget for overview in detailed mode
                max_overview_budget = remaining_budget * (0.2 if is_detailed_mode else 0.3)
                if overview_tokens <= max_overview_budget:
                    context_parts["repo_overview"] = repo_overview
                    remaining_budget -= overview_tokens
        
        # 2. Relevant modules (scale based on budget)
        if remaining_budget > 300:
            module_budget = budget * (0.4 if is_detailed_mode else 0.3)  # Use original budget
            modules_context = self._get_relevant_modules_context(analysis, module_budget)
            if modules_context:
                context_parts["relevant_modules"] = modules_context
                module_tokens = self.budget_manager.token_counter.count_tokens(modules_context)
                remaining_budget -= module_tokens
        
        # 3. Dependencies and relationships (more detail in detailed mode)
        if remaining_budget > 200:
            deps_budget = budget * (0.2 if is_detailed_mode else 0.3)  # Use original budget
            deps_context = self._get_dependency_context(analysis, deps_budget)
            if deps_context:
                context_parts["dependencies"] = deps_context
                deps_tokens = self.budget_manager.token_counter.count_tokens(deps_context)
                remaining_budget -= deps_tokens
        
        # 4. Targeted code samples (much more in detailed mode)
        if remaining_budget > 200:
            # Use much more budget for code samples in detailed mode
            samples_budget = min(remaining_budget, budget * (0.3 if is_detailed_mode else 0.2))
            code_samples = self._get_targeted_code_samples(query, analysis, samples_budget)
            if code_samples:
                context_parts["code_samples"] = code_samples
                samples_tokens = self.budget_manager.token_counter.count_tokens(code_samples)
                remaining_budget -= samples_tokens
                
        # 5. Additional context for detailed mode
        if is_detailed_mode and remaining_budget > 1000:
            # Add file contents for key files like cli.py
            file_content = self._get_key_file_contents(analysis, remaining_budget)
            if file_content:
                context_parts["file_contents"] = file_content
        
        return {
            "context_parts": context_parts,
            "total_tokens": budget - remaining_budget,
            "budget_used": ((budget - remaining_budget) / budget) * 100,
            "retrieval_strategy": "architectural"
        }
    
    def _get_implementation_context(
        self,
        query: str,
        analysis: QueryAnalysis,
        budget: int
    ) -> Dict[str, Any]:
        """Get implementation context prioritizing code samples and details."""
        
        context_parts = {}
        remaining_budget = budget
        
        # 1. Direct symbol/function implementations (highest priority)
        if analysis.symbols and remaining_budget > 400:
            impl_context = self._get_symbol_implementations(analysis.symbols, remaining_budget * 0.5)
            if impl_context:
                context_parts["implementations"] = impl_context
                impl_tokens = self.budget_manager.token_counter.count_tokens(impl_context)
                remaining_budget -= impl_tokens
        
        # 2. Related code examples
        if remaining_budget > 300:
            examples = self._get_code_examples(query, analysis, remaining_budget * 0.6)
            if examples:
                context_parts["examples"] = examples
                example_tokens = self.budget_manager.token_counter.count_tokens(examples)
                remaining_budget -= example_tokens
        
        # 3. Minimal architectural context
        if remaining_budget > 200:
            arch_context = self._get_minimal_architectural_context(analysis, remaining_budget)
            if arch_context:
                context_parts["architecture"] = arch_context
        
        return {
            "context_parts": context_parts,
            "total_tokens": budget - remaining_budget,
            "budget_used": ((budget - remaining_budget) / budget) * 100,
            "retrieval_strategy": "implementation"
        }
    
    def _get_usage_context(
        self,
        query: str,
        analysis: QueryAnalysis,
        budget: int
    ) -> Dict[str, Any]:
        """Get usage context prioritizing examples and public APIs."""
        
        context_parts = {}
        remaining_budget = budget
        
        # 1. Usage examples (highest priority)
        if remaining_budget > 400:
            usage_examples = self._get_usage_examples(query, analysis, remaining_budget * 0.6)
            if usage_examples:
                context_parts["usage_examples"] = usage_examples
                usage_tokens = self.budget_manager.token_counter.count_tokens(usage_examples)
                remaining_budget -= usage_tokens
        
        # 2. API documentation
        if remaining_budget > 200:
            api_context = self._get_api_context(analysis, remaining_budget * 0.4)
            if api_context:
                context_parts["api_docs"] = api_context
                api_tokens = self.budget_manager.token_counter.count_tokens(api_context)
                remaining_budget -= api_tokens
        
        # 3. Related patterns
        if remaining_budget > 100:
            patterns = self._get_usage_patterns(analysis, remaining_budget)
            if patterns:
                context_parts["patterns"] = patterns
        
        return {
            "context_parts": context_parts,
            "total_tokens": budget - remaining_budget,
            "budget_used": ((budget - remaining_budget) / budget) * 100,
            "retrieval_strategy": "usage"
        }
    
    def _get_general_context(
        self,
        query: str,
        analysis: QueryAnalysis,
        budget: int
    ) -> Dict[str, Any]:
        """Get balanced general context for queries without clear intent."""
        
        # Use hybrid retrieval if available, otherwise fall back to base retriever
        if self.hybrid_retriever:
            results = self.hybrid_retriever.retrieve(
                query=query,
                n_results=10,  # Get more initially for filtering
                use_code_context=True
            )
        else:
            results = self.base_retriever.retrieve_chunks(
                query=query,
                n_results=8
            )
        
        # Budget-aware context assembly
        context_parts = {}
        remaining_budget = budget
        
        if results:
            # Prioritize and truncate results to fit budget
            prioritized_results = self._prioritize_results(results, analysis)
            
            assembled_context = self._assemble_results_with_budget(
                prioritized_results, 
                remaining_budget
            )
            
            if assembled_context:
                context_parts["search_results"] = assembled_context
                result_tokens = self.budget_manager.token_counter.count_tokens(assembled_context)
                remaining_budget -= result_tokens
        
        # Add minimal repo context if budget allows
        if remaining_budget > 200 and self.repo_map:
            minimal_context = self._get_minimal_repo_context(analysis, remaining_budget)
            if minimal_context:
                context_parts["repo_context"] = minimal_context
        
        return {
            "context_parts": context_parts,
            "total_tokens": budget - remaining_budget,
            "budget_used": ((budget - remaining_budget) / budget) * 100,
            "retrieval_strategy": "general"
        }
    
    def _get_compact_repo_overview(self) -> Optional[str]:
        """Get a compact repository overview."""
        if not self.repo_map:
            return None
        
        try:
            overview = self.repo_map.get_module_overview()
            stats = overview.get("statistics", {})
            
            overview_parts = [
                f"Repository Statistics:",
                f"- Total files: {stats.get('total_files', 0)}",
                f"- Languages: {', '.join(stats.get('languages', {}).keys())}",
                f"- Main modules: {', '.join(stats.get('module_sizes', {}).keys())}"
            ]
            
            return "\n".join(overview_parts)
        except Exception:
            return None
    
    def _get_relevant_modules_context(self, analysis: QueryAnalysis, budget: int) -> Optional[str]:
        """Get context for modules relevant to the query."""
        if not self.repo_map:
            return None
        
        # Find modules that might contain the symbols mentioned in query
        relevant_modules = []
        
        for symbol in analysis.symbols:
            if self.code_index:
                # Find files containing this symbol
                symbol_locs = self.code_index.search_symbol(symbol)
                for loc in symbol_locs:
                    module_path = loc.get("filepath", "")
                    if module_path not in relevant_modules:
                        relevant_modules.append(module_path)
        
        # If no symbols found, search by file names and concepts
        if not relevant_modules and self.code_index:
            # Search for files with relevant names for architectural queries
            concept_keywords = ["cli", "main", "command", "app", "interface", "entry"]
            
            for file_path, metadata in self.code_index.file_index.items():
                file_name = file_path.lower()
                # Check if filename contains relevant concepts
                for keyword in concept_keywords:
                    if keyword in file_name:
                        relevant_modules.append(file_path)
                        break
                # Check if the file has CLI-related functions
                functions = metadata.get('functions', [])
                for func in functions:
                    if any(keyword in func.lower() for keyword in ['command', 'cli', 'main', 'app']):
                        if file_path not in relevant_modules:
                            relevant_modules.append(file_path)
                        break
        
        if not relevant_modules:
            return None
        
        # Create context for relevant modules
        module_parts = []
        for module_path in relevant_modules[:3]:  # Limit to top 3 modules
            if self.code_index:
                metadata = self.code_index.get_file_metadata(module_path)
                if metadata:
                    functions = metadata.get('functions', [])
                    classes = metadata.get('classes', [])
                    
                    # Include function names for CLI files since they're likely important
                    if 'cli.py' in module_path.lower() and len(functions) > 0:
                        # For detailed mode, show many more functions
                        is_detailed = budget > 20000  # Check if we have a large budget
                        max_functions = 20 if is_detailed else 8
                        
                        func_list = ', '.join(functions[:max_functions])
                        if len(functions) > max_functions:
                            func_list += f", and {len(functions) - max_functions} more"
                        
                        # In detailed mode, add more information
                        if is_detailed:
                            imports = metadata.get('imports', [])
                            import_summary = f", imports from {len(imports)} modules" if imports else ""
                            
                            module_parts.append(
                                f"- {module_path}: {len(functions)} functions ({func_list}){import_summary}, "
                                f"{len(classes)} classes"
                            )
                        else:
                            module_parts.append(
                                f"- {module_path}: {len(functions)} functions ({func_list}), "
                                f"{len(classes)} classes"
                            )
                    else:
                        module_parts.append(
                            f"- {module_path}: {len(functions)} functions, "
                            f"{len(classes)} classes"
                        )
        
        if module_parts:
            return "Relevant modules:\n" + "\n".join(module_parts)
        
        return None
    
    def _get_dependency_context(self, analysis: QueryAnalysis, budget: int) -> Optional[str]:
        """Get dependency context for the query."""
        if not self.repo_map or not analysis.symbols:
            return None
        
        # Get dependency information for symbols
        deps_info = []
        
        for symbol in analysis.symbols[:2]:  # Limit to avoid budget overflow
            if self.code_index:
                symbol_locs = self.code_index.search_symbol(symbol)
                for loc in symbol_locs[:1]:  # One location per symbol
                    filepath = loc.get("filepath", "")
                    related_files = self.repo_map.get_related_files(filepath)
                    if related_files:
                        deps_info.append(f"- {symbol} (in {filepath}) relates to: {', '.join(related_files[:3])}")
        
        if deps_info:
            return "Dependencies:\n" + "\n".join(deps_info)
        
        return None
    
    def _get_targeted_code_samples(
        self,
        query: str,
        analysis: QueryAnalysis,
        budget: int
    ) -> Optional[str]:
        """Get targeted code samples based on query analysis."""
        
        if not self.hybrid_retriever:
            return None
        
        # Get code examples using hybrid retrieval
        try:
            examples = self.hybrid_retriever.retrieve_code_examples(
                query=query,
                n_results=3
            )
            
            if not examples:
                return None
            
            sample_parts = []
            used_tokens = 0
            
            for example in examples:
                content = example.get("content", "")
                
                # For detailed mode, allow much longer examples
                is_detailed = budget > 20000
                max_length = 2000 if is_detailed else 500
                
                # Truncate long examples based on mode
                if len(content) > max_length:
                    content = content[:max_length] + "\n..."
                
                tokens_needed = self.budget_manager.token_counter.count_tokens(content)
                if used_tokens + tokens_needed > budget:
                    break
                
                filepath = example.get("metadata", {}).get("document_id", "unknown")
                sample_parts.append(f"From {filepath}:\n```\n{content}\n```")
                used_tokens += tokens_needed
            
            if sample_parts:
                return "Code samples:\n" + "\n\n".join(sample_parts)
        
        except Exception:
            pass
        
        return None
    
    def _get_symbol_implementations(self, symbols: List[str], budget: int) -> Optional[str]:
        """Get implementations for specific symbols."""
        if not self.code_index:
            return None
        
        impl_parts = []
        used_tokens = 0
        
        for symbol in symbols:
            if used_tokens >= budget * 0.8:  # Leave some budget for formatting
                break
            
            # Try function search first, then symbol search
            locations = self.code_index.search_function(symbol)
            if not locations:
                locations = self.code_index.search_symbol(symbol)
            
            for loc in locations[:1]:  # One implementation per symbol
                filepath = loc.get("filepath", "")
                line = loc.get("line", "unknown")
                
                # Try to get actual implementation
                try:
                    impl_context = f"- {symbol} in {filepath}:{line}"
                    tokens_needed = self.budget_manager.token_counter.count_tokens(impl_context)
                    
                    if used_tokens + tokens_needed <= budget:
                        impl_parts.append(impl_context)
                        used_tokens += tokens_needed
                except Exception:
                    continue
        
        if impl_parts:
            return "Symbol implementations:\n" + "\n".join(impl_parts)
        
        return None
    
    def _get_code_examples(self, query: str, analysis: QueryAnalysis, budget: int) -> Optional[str]:
        """Get code examples relevant to the query."""
        return self._get_targeted_code_samples(query, analysis, budget)
    
    def _get_minimal_architectural_context(self, analysis: QueryAnalysis, budget: int) -> Optional[str]:
        """Get minimal architectural context."""
        if not self.repo_map:
            return None
        
        # Just basic module categorization
        if analysis.symbols and self.code_index:
            arch_parts = []
            for symbol in analysis.symbols[:2]:
                symbol_locs = self.code_index.search_symbol(symbol)
                for loc in symbol_locs[:1]:
                    filepath = loc.get("filepath", "")
                    category = self.repo_map._categorize_module(filepath)
                    arch_parts.append(f"- {symbol}: {category} module")
            
            if arch_parts:
                return "Module categories:\n" + "\n".join(arch_parts)
        
        return None
    
    def _get_usage_examples(self, query: str, analysis: QueryAnalysis, budget: int) -> Optional[str]:
        """Get usage examples for the query."""
        # Similar to code examples but focused on usage patterns
        return self._get_targeted_code_samples(query, analysis, budget)
    
    def _get_api_context(self, analysis: QueryAnalysis, budget: int) -> Optional[str]:
        """Get API context for the query."""
        if not self.code_index or not analysis.symbols:
            return None
        
        api_parts = []
        for symbol in analysis.symbols[:3]:
            # Look for public functions/classes
            locations = self.code_index.search_function(symbol)
            if not locations:
                locations = self.code_index.search_class(symbol)
            
            for loc in locations[:1]:
                filepath = loc.get("filepath", "")
                api_parts.append(f"- {symbol} in {filepath}")
        
        if api_parts:
            return "API references:\n" + "\n".join(api_parts)
        
        return None
    
    def _get_usage_patterns(self, analysis: QueryAnalysis, budget: int) -> Optional[str]:
        """Get usage patterns for the query."""
        # Placeholder for usage pattern analysis
        return None
    
    def _get_minimal_repo_context(self, analysis: QueryAnalysis, budget: int) -> Optional[str]:
        """Get minimal repository context."""
        if not self.repo_map:
            return None
        
        try:
            stats = self.repo_map.stats
            context = f"Repository: {stats.get('total_files', 0)} files, {len(stats.get('languages', {}))} languages"
            return context
        except Exception:
            return None
    
    def _prioritize_results(self, results: List[Dict[str, Any]], analysis: QueryAnalysis) -> List[Dict[str, Any]]:
        """Prioritize search results based on query analysis."""
        
        # Simple prioritization: prefer results with symbols from query
        prioritized = []
        
        for result in results:
            priority_score = 0
            
            # Check if result contains query symbols
            content = result.get("content", "").lower()
            symbols = result.get("metadata", {}).get("symbols", [])
            
            for symbol in analysis.symbols:
                if symbol.lower() in content or symbol in symbols:
                    priority_score += 2
            
            # Prefer results with more metadata
            if result.get("metadata", {}).get("chunk_type"):
                priority_score += 1
            
            result["priority_score"] = priority_score
            prioritized.append(result)
        
        # Sort by priority score (higher first)
        return sorted(prioritized, key=lambda x: x.get("priority_score", 0), reverse=True)
    
    def _assemble_results_with_budget(
        self,
        results: List[Dict[str, Any]],
        budget: int
    ) -> Optional[str]:
        """Assemble search results within the given token budget."""
        
        if not results:
            return None
        
        assembled_parts = []
        used_tokens = 0
        
        for result in results:
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            
            # Format result with metadata
            formatted = self._format_search_result(content, metadata)
            
            tokens_needed = self.budget_manager.token_counter.count_tokens(formatted)
            
            if used_tokens + tokens_needed > budget:
                # Try to truncate if this is the first result and we have significant budget
                if not assembled_parts and budget > 200:
                    truncated = self.budget_manager.truncate_to_budget(formatted)
                    assembled_parts.append(truncated)
                break
            
            assembled_parts.append(formatted)
            used_tokens += tokens_needed
        
        if assembled_parts:
            return "\n\n---\n\n".join(assembled_parts)
        
        return None
    
    def _format_search_result(self, content: str, metadata: Dict[str, Any]) -> str:
        """Format a search result with metadata."""
        
        parts = []
        
        # Add file info if available
        doc_id = metadata.get("document_id", "")
        if doc_id:
            parts.append(f"File: {doc_id}")
        
        # Add line info if available
        start_line = metadata.get("start_line")
        end_line = metadata.get("end_line")
        if start_line and end_line:
            if start_line == end_line:
                parts.append(f"Line: {start_line}")
            else:
                parts.append(f"Lines: {start_line}-{end_line}")
        
        # Add chunk type if available
        chunk_type = metadata.get("chunk_type")
        if chunk_type:
            parts.append(f"Type: {chunk_type}")
        
        header = " | ".join(parts) if parts else "Code:"
        
        return f"{header}\n```\n{content}\n```"
    
    def _get_detailed_repo_overview(self) -> Optional[str]:
        """Get a detailed repository overview for high-budget contexts."""
        if not self.repo_map:
            return None
        
        try:
            overview = self.repo_map.get_module_overview()
            stats = overview.get("statistics", {})
            
            overview_parts = [
                f"Repository Overview:",
                f"- Total files: {stats.get('total_files', 0)}",
                f"- Languages: {', '.join(stats.get('languages', {}).keys())}",
                f"- Main modules: {', '.join(stats.get('module_sizes', {}).keys())}",
                "",
                "Key directories and their purposes:"
            ]
            
            # Add directory structure
            if self.code_index:
                directories = {}
                for file_path in self.code_index.file_index.keys():
                    dir_name = "/".join(file_path.split("/")[:-1])
                    if "core" in dir_name:
                        directories[dir_name] = directories.get(dir_name, 0) + 1
                
                for dir_name, file_count in sorted(directories.items())[:10]:
                    if file_count > 1:  # Only show directories with multiple files
                        overview_parts.append(f"- {dir_name}: {file_count} files")
            
            return "\n".join(overview_parts)
        except Exception:
            return None
    
    def _get_key_file_contents(self, analysis: QueryAnalysis, budget: int) -> Optional[str]:
        """Get actual file contents for key files in detailed mode."""
        if not self.code_index:
            return None
        
        # Find key files mentioned in the query or CLI-related files
        key_files = []
        
        # Look for CLI-related files
        for file_path in self.code_index.file_index.keys():
            if any(keyword in file_path.lower() for keyword in ['cli.py', 'main.py', 'app.py']):
                key_files.append(file_path)
        
        if not key_files:
            return None
        
        content_parts = []
        used_budget = 0
        
        for file_path in key_files[:2]:  # Limit to 2 files to avoid budget overflow
            try:
                # Get file metadata to show structure
                metadata = self.code_index.get_file_metadata(file_path)
                if metadata:
                    functions = metadata.get('functions', [])
                    classes = metadata.get('classes', [])
                    
                    file_summary = [
                        f"File: {file_path}",
                        f"Functions ({len(functions)}): {', '.join(functions[:15])}{'...' if len(functions) > 15 else ''}",
                        f"Classes ({len(classes)}): {', '.join(classes[:10])}{'...' if len(classes) > 10 else ''}",
                        ""
                    ]
                    
                    file_content = "\n".join(file_summary)
                    content_tokens = self.budget_manager.token_counter.count_tokens(file_content)
                    
                    if used_budget + content_tokens <= budget:
                        content_parts.append(file_content)
                        used_budget += content_tokens
                    else:
                        break
                        
            except Exception:
                continue
        
        if content_parts:
            return "Key file details:\n\n" + "\n".join(content_parts)
        
        return None