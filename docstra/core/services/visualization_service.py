"""
Service for generating visualizations of repository structure and metrics.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from pathlib import Path

from rich.console import Console

from docstra.core.indexing.code_index import CodebaseIndex
from docstra.core.indexing.repo_map import RepositoryMap
from docstra.core.utils.colors import Colors


class VisualizationService:
    """Service for generating repository visualizations."""

    def __init__(self, repo_map: RepositoryMap, code_index: CodebaseIndex, console: Optional[Console] = None):
        """Initialize the visualization service.
        
        Args:
            repo_map: Repository map
            code_index: Code index
            console: Optional console for output
        """
        self.repo_map = repo_map
        self.code_index = code_index
        self.console = console or Console()

    def generate_dependency_graph(self, output_file: Optional[str] = None) -> str:
        """Generate a dependency graph visualization.
        
        Args:
            output_file: Optional output file path
            
        Returns:
            Path to generated visualization or description
        """
        try:
            # For now, generate a simple text-based dependency representation
            dependencies = self.repo_map.stats.get("dependencies", {})
            
            if not dependencies:
                return "No dependencies found to visualize"
            
            # Create a simple DOT format graph
            dot_content = ["digraph Dependencies {"]
            dot_content.append("    rankdir=LR;")
            dot_content.append("    node [shape=box];")
            
            # Add nodes and edges
            for file_path, deps in dependencies.items():
                file_name = os.path.basename(file_path)
                for dep in deps:
                    dep_name = os.path.basename(dep)
                    dot_content.append(f'    "{file_name}" -> "{dep_name}";')
            
            dot_content.append("}")
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write('\n'.join(dot_content))
                return output_file
            else:
                # Return the content for display
                return '\n'.join(dot_content[:20])  # First 20 lines
                
        except Exception as e:
            return f"Error generating dependency graph: {e}"

    def generate_complexity_heatmap(self, output_file: Optional[str] = None) -> str:
        """Generate a complexity heatmap visualization.
        
        Args:
            output_file: Optional output file path
            
        Returns:
            Path to generated visualization or description
        """
        try:
            complexity_data = self.repo_map.stats.get("complexity", {})
            
            if not complexity_data:
                return "No complexity data found to visualize"
            
            # Create a simple text-based heatmap
            sorted_files = sorted(complexity_data.items(), key=lambda x: x[1], reverse=True)
            
            heatmap_content = ["# Complexity Heatmap"]
            heatmap_content.append("# Format: File | Complexity | Visual")
            heatmap_content.append("")
            
            for file_path, complexity in sorted_files[:20]:  # Top 20 most complex
                file_name = os.path.basename(file_path)
                # Create a simple bar visualization
                bar_length = min(complexity // 2, 20)  # Scale complexity to bar length
                bar = "█" * bar_length
                heatmap_content.append(f"{file_name:<30} | {complexity:>3} | {bar}")
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write('\n'.join(heatmap_content))
                return output_file
            else:
                return '\n'.join(heatmap_content)
                
        except Exception as e:
            return f"Error generating complexity heatmap: {e}"

    def generate_architecture_diagram(self, output_file: Optional[str] = None) -> str:
        """Generate an architecture diagram.
        
        Args:
            output_file: Optional output file path
            
        Returns:
            Path to generated visualization or description
        """
        try:
            overview = self.repo_map.get_module_overview()
            modules = overview.get("modules", {})
            
            if not modules:
                return "No module information found to visualize"
            
            # Create a simple architecture representation
            arch_content = ["# Architecture Overview"]
            arch_content.append("")
            
            for category, files in modules.items():
                arch_content.append(f"## {category.title()} Module")
                arch_content.append(f"Files: {len(files)}")
                
                # Show languages used
                languages = list(set(f.get("language") for f in files if f.get("language")))
                if languages:
                    arch_content.append(f"Languages: {', '.join(languages)}")
                
                # Show key files (those with most symbols)
                files_with_symbols = [(f.get("path", ""), len(f.get("symbols", []))) for f in files]
                top_files = sorted(files_with_symbols, key=lambda x: x[1], reverse=True)[:3]
                
                if top_files:
                    arch_content.append("Key files:")
                    for file_path, symbol_count in top_files:
                        if file_path:
                            arch_content.append(f"  - {os.path.basename(file_path)} ({symbol_count} symbols)")
                
                arch_content.append("")
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write('\n'.join(arch_content))
                return output_file
            else:
                return '\n'.join(arch_content)
                
        except Exception as e:
            return f"Error generating architecture diagram: {e}"

    def create_interactive_explorer(self, output_dir: str = "docstra_viz") -> str:
        """Create an interactive web-based repository explorer.
        
        Args:
            output_dir: Directory to create the interactive explorer
            
        Returns:
            Path to the generated explorer
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate a simple HTML page with repository overview
            html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Repository Explorer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .module { border: 1px solid #ccc; margin: 10px; padding: 10px; }
        .stats { background-color: #f5f5f5; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Repository Explorer</h1>
"""
            
            # Add repository statistics
            stats = self.repo_map.stats
            html_content += f"""
    <div class="stats">
        <h2>Repository Statistics</h2>
        <p>Total Files: {stats.get('total_files', 0)}</p>
        <p>Total Lines: {stats.get('total_lines', 0):,}</p>
        <p>Languages: {len(stats.get('languages', {}))}</p>
    </div>
"""
            
            # Add module information
            overview = self.repo_map.get_module_overview()
            modules = overview.get("modules", {})
            
            html_content += "<h2>Modules</h2>"
            for category, files in modules.items():
                html_content += f"""
    <div class="module">
        <h3>{category.title()}</h3>
        <p>Files: {len(files)}</p>
        <ul>
"""
                for file_info in files[:5]:  # Show first 5 files
                    file_path = file_info.get("path", "")
                    if file_path:
                        html_content += f"<li>{os.path.basename(file_path)}</li>"
                
                if len(files) > 5:
                    html_content += f"<li><em>... and {len(files) - 5} more files</em></li>"
                
                html_content += """
        </ul>
    </div>
"""
            
            html_content += """
</body>
</html>
"""
            
            index_path = os.path.join(output_dir, "index.html")
            with open(index_path, 'w') as f:
                f.write(html_content)
            
            return index_path
            
        except Exception as e:
            return f"Error creating interactive explorer: {e}" 