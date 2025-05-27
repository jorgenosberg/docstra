"""
Advanced prompt templates for documentation generation.

This module contains sophisticated prompt templates designed to generate high-quality
documentation using LLMs with rich contextual information.
"""

from typing import Dict, Any, Optional


class EnhancedDocumentationPrompts:
    """Enhanced prompt templates for different documentation types with rich context integration."""
    
    # Project overview prompt with comprehensive context
    PROJECT_OVERVIEW_PROMPT = """You are an expert technical writer creating comprehensive project documentation.

# Project Overview Request

## Project Information
- Name: {project_name}
- Description: {project_description}
- Root Path: {project_path}
- Total Files: {total_files}
- Total Lines: {total_lines:,}
- Languages: {languages}

## Codebase Structure
{repo_structure}

## Module Overview
{module_overview}

## Key Statistics
{statistics}

## Repository Context
{repository_context}

## Task
Generate a comprehensive project overview that serves as the main documentation entry point. Include:

1. **Executive Summary**: What this project does and why it matters
2. **Architecture Overview**: High-level system design and key components
3. **Getting Started**: How developers can understand and work with this codebase
4. **Module Guide**: Brief description of each major module/component
5. **Development Workflow**: How the codebase is organized and typical development patterns
6. **Key Features**: Main capabilities and functionality
7. **Technology Stack**: Languages, frameworks, and tools used

## Documentation Style Guidelines
- Use clear, professional language appropriate for developers
- Start with a compelling overview that explains the project's purpose
- Organize content with logical headers and sections
- Include practical information for getting started
- Use bullet points and lists for clarity
- Cross-reference modules and components
- Focus on helping new developers understand the project quickly
- Write in a professional yet accessible tone
- Avoid implementation details - focus on conceptual understanding

{style_instructions}
"""

    # Module overview prompt with rich context
    MODULE_OVERVIEW_PROMPT = """You are an expert technical writer creating module documentation.

# Module Documentation Request

## Module Information
- Module: {module_name}
- Category: {module_category}
- Files Count: {files_count}
- Total Lines: {total_lines:,}

## Files in This Module
{files_list}

## Module Dependencies
{dependencies}

## Related Modules
{related_modules}

## Key Symbols/Components
{key_symbols}

## Module Context
{module_context}

## Task
Generate comprehensive module documentation that includes:

1. **Module Purpose**: What this module does and its role in the larger system
2. **Key Components**: Main classes, functions, and important files
3. **Architecture**: How components within the module work together
4. **Usage Patterns**: Common ways this module is used
5. **Dependencies**: What this module depends on and what depends on it
6. **Important Files**: Brief description of the most important files
7. **Public API**: Main interfaces exposed by this module
8. **Integration Points**: How this module connects with other parts of the system

## Documentation Style Guidelines
- Focus on the module's role in the system architecture
- Help developers understand when and how to use this module
- Use clear organization with proper headers and formatting
- Include practical usage examples where appropriate
- Explain relationships and dependencies clearly
- Highlight important design patterns or architectural decisions

{style_instructions}
"""

    # File documentation prompt with comprehensive context
    FILE_DOCUMENTATION_PROMPT = """You are an expert technical writer creating detailed code documentation.

# File Documentation Request

## File Information
- Path: {filepath}
- Language: {language}
- Lines: {line_count}
- Size: {file_size} bytes
- Module Category: {module_category}

## Code Structure
- Classes: {classes}
- Functions: {functions}
- Imports: {imports}

## File Context
{file_context}

## Related Files
{related_files}

## Dependencies
{dependencies}

## Similar Code Examples
{similar_examples}

## Cross References
{cross_references}

## Code to Document
```{language}
{code_content}
```

## Task
Generate comprehensive documentation for this file that includes:

1. **Overview**: What this file does and its purpose in the system
2. **Key Components**: Main classes and functions with their responsibilities
3. **Architecture**: How components in this file work together
4. **Usage Examples**: How other parts of the codebase use this file
5. **Implementation Details**: Important algorithms, patterns, or design decisions
6. **Dependencies**: What this file depends on and relationships to other files
7. **API Reference**: Detailed documentation of public interfaces
8. **Error Handling**: How errors are handled and propagated
9. **Performance Considerations**: Any important performance notes
10. **Testing**: How to test the functionality in this file

## Documentation Style Guidelines
- Start with a clear overview paragraph explaining the file's purpose
- Use proper markdown formatting with headers
- Include code examples where helpful for understanding usage
- Explain complex logic and algorithms step by step
- Document public APIs thoroughly with parameters and return values
- Mention important edge cases, limitations, or gotchas
- Cross-reference related files and components using relative paths
- Use consistent terminology throughout
- Focus on helping developers understand both WHAT the code does and HOW it fits into the larger system
- Be thorough but concise, avoiding unnecessary verbosity
- Use professional technical writing style

{style_instructions}
"""

    # User guide prompt
    USER_GUIDE_PROMPT = """You are an expert technical writer creating user guides.

# Guide Generation Request

## Guide Information
- Title: {guide_title}
- Guide Type: {guide_type}
- Project: {project_name}
- Description: {project_description}

## Project Context
{repository_context}

## Target Audience
{target_audience}

## Task
Generate a comprehensive {guide_title} that includes:

1. **Introduction**: Brief overview of what this guide covers
2. **Prerequisites**: What users need before starting
3. **Step-by-step Instructions**: Clear, actionable steps
4. **Examples**: Practical examples and code snippets
5. **Common Issues**: Potential problems and solutions
6. **Next Steps**: What to do after completing this guide
7. **Additional Resources**: Links to related documentation

## Documentation Style Guidelines
- Focus on practical, actionable guidance
- Use clear language appropriate for the target audience
- Provide complete, working examples
- Organize content logically with proper headers and formatting
- Include troubleshooting information
- Use numbered lists for sequential steps
- Include code blocks with proper syntax highlighting

{style_instructions}
"""

    # API documentation prompt
    API_DOCUMENTATION_PROMPT = """You are an expert technical writer creating API documentation.

# API Documentation Request

## Project Information
- Name: {project_name}
- Description: {project_description}
- API Files: {api_files_count}

## Public Classes
{public_classes}

## Public Functions  
{public_functions}

## API Context
{api_context}

## Task
Generate a comprehensive API overview that includes:

1. **API Introduction**: Overview of the API and its purpose
2. **Getting Started**: How to use the API
3. **Authentication**: How to authenticate (if applicable)
4. **Core Concepts**: Key concepts developers need to understand
5. **Class Reference**: Overview of main classes and their purposes
6. **Function Reference**: Overview of main functions and utilities
7. **Examples**: Common usage patterns and examples
8. **Error Handling**: How errors are handled in the API
9. **Rate Limits**: Any rate limiting information (if applicable)
10. **Versioning**: API version information

## Documentation Style Guidelines
- Focus on helping developers understand the API structure and usage patterns
- Organize content logically with clear navigation
- Include practical examples for common use cases
- Document all public interfaces thoroughly
- Use consistent formatting for code examples
- Include both simple and complex usage examples

{style_instructions}
"""

    @classmethod
    def format_project_overview_prompt(cls, **kwargs) -> str:
        """Format the project overview prompt with provided context."""
        # Provide defaults for optional parameters
        defaults = {
            'total_lines': 0,
            'repo_structure': 'Not available',
            'module_overview': 'Not available', 
            'statistics': 'Not available',
            'repository_context': 'Not available',
            'style_instructions': ''
        }
        defaults.update(kwargs)
        return cls.PROJECT_OVERVIEW_PROMPT.format(**defaults)
    
    @classmethod
    def format_module_overview_prompt(cls, **kwargs) -> str:
        """Format the module overview prompt with provided context."""
        defaults = {
            'module_category': 'unknown',
            'files_list': 'No files listed',
            'dependencies': 'None identified',
            'related_modules': 'None identified',
            'key_symbols': 'No symbols identified',
            'module_context': 'Not available',
            'style_instructions': ''
        }
        defaults.update(kwargs)
        return cls.MODULE_OVERVIEW_PROMPT.format(**defaults)
    
    @classmethod
    def format_file_documentation_prompt(cls, **kwargs) -> str:
        """Format the file documentation prompt with provided context."""
        defaults = {
            'module_category': 'unknown',
            'classes': 'None',
            'functions': 'None', 
            'imports': 'None',
            'file_context': 'No additional context available',
            'related_files': 'None',
            'dependencies': 'None',
            'similar_examples': 'No similar examples found',
            'cross_references': 'None',
            'style_instructions': ''
        }
        defaults.update(kwargs)
        return cls.FILE_DOCUMENTATION_PROMPT.format(**defaults)

    @classmethod
    def format_user_guide_prompt(cls, **kwargs) -> str:
        """Format the user guide prompt with provided context."""
        defaults = {
            'guide_type': 'user guide',
            'repository_context': 'Not available',
            'target_audience': 'developers',
            'style_instructions': ''
        }
        defaults.update(kwargs)
        return cls.USER_GUIDE_PROMPT.format(**defaults)

    @classmethod
    def format_api_documentation_prompt(cls, **kwargs) -> str:
        """Format the API documentation prompt with provided context."""
        defaults = {
            'api_files_count': 0,
            'public_classes': 'None identified',
            'public_functions': 'None identified',
            'api_context': 'Not available',
            'style_instructions': ''
        }
        defaults.update(kwargs)
        return cls.API_DOCUMENTATION_PROMPT.format(**defaults)


class PromptFormatters:
    """Utility functions for formatting context information for prompts."""
    
    @staticmethod
    def format_file_list(files: list, max_files: int = 10) -> str:
        """Format a list of files for inclusion in prompts."""
        if not files:
            return "No files found"
        
        formatted = []
        for i, file in enumerate(files[:max_files]):
            if hasattr(file, 'metadata'):
                # Document object
                formatted.append(f"- **{file.metadata.filepath}** ({file.metadata.line_count} lines)")
            elif isinstance(file, dict):
                # Dictionary with file info
                formatted.append(f"- **{file.get('path', 'unknown')}** ({file.get('lines', '?')} lines)")
            else:
                # String path
                formatted.append(f"- **{file}**")
        
        if len(files) > max_files:
            formatted.append(f"- ... and {len(files) - max_files} more files")
        
        return "\n".join(formatted)
    
    @staticmethod
    def format_dependencies(dependencies: list, max_deps: int = 8) -> str:
        """Format a list of dependencies for inclusion in prompts."""
        if not dependencies:
            return "None identified"
        
        formatted = [f"- {dep}" for dep in dependencies[:max_deps]]
        if len(dependencies) > max_deps:
            formatted.append(f"- ... and {len(dependencies) - max_deps} more dependencies")
        
        return "\n".join(formatted)
    
    @staticmethod
    def format_symbols(symbols: list, symbol_type: str = "symbols", max_symbols: int = 10) -> str:
        """Format a list of symbols (classes, functions, etc.) for inclusion in prompts."""
        if not symbols:
            return f"No {symbol_type} found"
        
        if len(symbols) <= max_symbols:
            return f"{', '.join(symbols)}"
        else:
            return f"{', '.join(symbols[:max_symbols])}, and {len(symbols) - max_symbols} more"
    
    @staticmethod
    def format_similar_examples(examples: list, max_examples: int = 3) -> str:
        """Format similar code examples for inclusion in prompts."""
        if not examples:
            return "No similar examples found"
        
        formatted = []
        for example in examples[:max_examples]:
            if isinstance(example, dict):
                filepath = example.get('filepath', 'unknown')
                content = example.get('content', '')[:200]  # Limit content length
                relevance = example.get('relevance', 'unknown')
                formatted.append(f"**{filepath}** (relevance: {relevance}):\n```\n{content}...\n```")
            else:
                formatted.append(str(example))
        
        return "\n\n".join(formatted)
    
    @staticmethod
    def format_cross_references(cross_refs: list, max_refs: int = 5) -> str:
        """Format cross-references for inclusion in prompts."""
        if not cross_refs:
            return "None identified"
        
        formatted = []
        for ref in cross_refs[:max_refs]:
            if isinstance(ref, dict):
                ref_type = ref.get('type', 'reference')
                filepath = ref.get('filepath', 'unknown')
                context = ref.get('context', '')[:100]  # Limit context length
                formatted.append(f"- **{ref_type.title()}** in {filepath}: {context}...")
            else:
                formatted.append(f"- {ref}")
        
        if len(cross_refs) > max_refs:
            formatted.append(f"- ... and {len(cross_refs) - max_refs} more references")
        
        return "\n".join(formatted)

    @staticmethod
    def get_style_instructions(style_guide: Optional[str] = None) -> str:
        """Get formatted style instructions."""
        base_style = """
## Style Guidelines
- Use clear, professional language
- Include practical examples where appropriate
- Organize content with logical headers
- Use code blocks for technical examples
- Cross-reference related components
- Focus on developer needs and use cases
"""
        if style_guide:
            return base_style + f"\n## Custom Style Guide\n{style_guide}"
        return base_style 