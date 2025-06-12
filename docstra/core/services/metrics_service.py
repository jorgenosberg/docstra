"""
Service for calculating comprehensive code metrics.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from docstra.core.indexing.code_index import CodebaseIndex
from docstra.core.indexing.repo_map import RepositoryMap
from docstra.core.utils.colors import Colors


class MetricsService:
    """Service for calculating comprehensive code metrics."""

    def __init__(self, repo_map: RepositoryMap, code_index: CodebaseIndex, console: Optional[Console] = None):
        """Initialize the metrics service.
        
        Args:
            repo_map: Repository map
            code_index: Code index
            console: Optional console for output
        """
        self.repo_map = repo_map
        self.code_index = code_index
        self.console = console or Console()

    def calculate_repository_overview(self) -> Dict[str, Any]:
        """Calculate high-level repository metrics.
        
        Returns:
            Dictionary containing repository overview
        """
        overview = self.repo_map.get_module_overview()
        stats = overview.get("statistics", {})

        return {
            "repository_stats": {
                "total_files": stats.get("total_files", 0),
                "total_lines": stats.get("total_lines", 0),
                "languages": stats.get("languages", {}),
                "modules": len(overview.get("modules", {}))
            },
            "complexity_analysis": self._analyze_overall_complexity(),
            "dependency_analysis": self._analyze_dependencies(),
            "module_breakdown": self._analyze_module_breakdown(overview)
        }

    def _analyze_overall_complexity(self) -> Dict[str, Any]:
        """Analyze overall complexity metrics.
        
        Returns:
            Dictionary containing complexity analysis
        """
        complexity_data = self.repo_map.stats.get("complexity", {})
        
        if not complexity_data:
            return {"total_files": 0, "average_complexity": 0, "max_complexity": 0}

        complexities = list(complexity_data.values())
        
        return {
            "total_files": len(complexities),
            "average_complexity": sum(complexities) / len(complexities),
            "max_complexity": max(complexities),
            "min_complexity": min(complexities),
            "high_complexity_files": len([c for c in complexities if c > 10]),
            "complexity_distribution": self._get_complexity_distribution(complexities)
        }

    def _get_complexity_distribution(self, complexities: List[int]) -> Dict[str, int]:
        """Get distribution of complexity scores.
        
        Args:
            complexities: List of complexity scores
            
        Returns:
            Dictionary with complexity distribution
        """
        distribution = {
            "low (1-5)": 0,
            "medium (6-10)": 0,
            "high (11-20)": 0,
            "very_high (20+)": 0
        }

        for complexity in complexities:
            if complexity <= 5:
                distribution["low (1-5)"] += 1
            elif complexity <= 10:
                distribution["medium (6-10)"] += 1
            elif complexity <= 20:
                distribution["high (11-20)"] += 1
            else:
                distribution["very_high (20+)"] += 1

        return distribution

    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependency relationships.
        
        Returns:
            Dictionary containing dependency analysis
        """
        dependencies = self.repo_map.stats.get("dependencies", {})
        
        total_dependencies = sum(len(deps) for deps in dependencies.values())
        avg_dependencies = total_dependencies / len(dependencies) if dependencies else 0

        # Find highly coupled files
        highly_coupled = [
            {"file": file_path, "dependency_count": len(deps)}
            for file_path, deps in dependencies.items()
            if len(deps) > 5
        ]

        return {
            "total_files_with_deps": len(dependencies),
            "total_dependencies": total_dependencies,
            "average_dependencies_per_file": avg_dependencies,
            "highly_coupled_files": sorted(highly_coupled, key=lambda x: x["dependency_count"], reverse=True)[:10],
            "dependency_cycles": self.detect_dependency_cycles()
        }

    def _analyze_module_breakdown(self, overview: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze breakdown by module categories.
        
        Args:
            overview: Module overview data
            
        Returns:
            Dictionary containing module breakdown
        """
        modules = overview.get("modules", {})
        
        breakdown = {}
        for category, files in modules.items():
            breakdown[category] = {
                "file_count": len(files),
                "languages": list(set(f.get("language") for f in files if f.get("language"))),
                "total_symbols": sum(len(f.get("symbols", [])) for f in files),
                "avg_symbols_per_file": sum(len(f.get("symbols", [])) for f in files) / len(files) if files else 0
            }

        return breakdown

    def detect_dependency_cycles(self) -> List[List[str]]:
        """Detect circular dependencies in the codebase.
        
        Returns:
            List of dependency cycles
        """
        dependencies = self.repo_map.stats.get("dependencies", {})
        
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        recursion_stack: Set[str] = set()

        def dfs_detect_cycle(node: str, path: List[str]) -> None:
            if node in recursion_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            recursion_stack.add(node)

            for dependency in dependencies.get(node, []):
                if dependency in dependencies:  # Only follow if dependency is also in our map
                    dfs_detect_cycle(dependency, path + [node])

            recursion_stack.remove(node)

        for file_path in dependencies:
            if file_path not in visited:
                dfs_detect_cycle(file_path, [])

        return cycles

    def analyze_coupling(self) -> Dict[str, Any]:
        """Analyze module coupling metrics.
        
        Returns:
            Dictionary containing coupling analysis
        """
        dependencies = self.repo_map.stats.get("dependencies", {})
        
        # Calculate coupling metrics
        afferent_coupling = {}  # How many files depend on this file
        efferent_coupling = {}  # How many files this file depends on

        # Initialize
        for file_path in dependencies:
            afferent_coupling[file_path] = 0
            efferent_coupling[file_path] = len(dependencies[file_path])

        # Calculate afferent coupling
        for file_path, deps in dependencies.items():
            for dep in deps:
                if dep in afferent_coupling:
                    afferent_coupling[dep] += 1

        # Calculate instability (Ce / (Ca + Ce))
        instability = {}
        for file_path in dependencies:
            ca = afferent_coupling[file_path]  # Afferent coupling
            ce = efferent_coupling[file_path]  # Efferent coupling
            if ca + ce > 0:
                instability[file_path] = ce / (ca + ce)
            else:
                instability[file_path] = 0

        return {
            "afferent_coupling": afferent_coupling,
            "efferent_coupling": efferent_coupling,
            "instability": instability,
            "most_stable_files": self._get_most_stable_files(instability),
            "most_unstable_files": self._get_most_unstable_files(instability)
        }

    def _get_most_stable_files(self, instability: Dict[str, float]) -> List[Dict[str, Any]]:
        """Get the most stable files (low instability).
        
        Args:
            instability: Instability scores
            
        Returns:
            List of most stable files
        """
        sorted_files = sorted(instability.items(), key=lambda x: x[1])[:10]
        return [
            {"file": os.path.basename(file_path), "instability": score}
            for file_path, score in sorted_files
        ]

    def _get_most_unstable_files(self, instability: Dict[str, float]) -> List[Dict[str, Any]]:
        """Get the most unstable files (high instability).
        
        Args:
            instability: Instability scores
            
        Returns:
            List of most unstable files
        """
        sorted_files = sorted(instability.items(), key=lambda x: x[1], reverse=True)[:10]
        return [
            {"file": os.path.basename(file_path), "instability": score}
            for file_path, score in sorted_files
        ]

    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate a comprehensive code quality report.
        
        Returns:
            Dictionary containing quality report
        """
        complexity_analysis = self._analyze_overall_complexity()
        dependency_analysis = self._analyze_dependencies()
        coupling_analysis = self.analyze_coupling()

        # Calculate quality scores
        quality_score = self._calculate_quality_score(complexity_analysis, dependency_analysis)

        return {
            "overall_quality_score": quality_score,
            "complexity_metrics": complexity_analysis,
            "dependency_metrics": dependency_analysis,
            "coupling_metrics": coupling_analysis,
            "recommendations": self._generate_recommendations(complexity_analysis, dependency_analysis, coupling_analysis)
        }

    def _calculate_quality_score(self, complexity_analysis: Dict[str, Any], dependency_analysis: Dict[str, Any]) -> float:
        """Calculate an overall quality score.
        
        Args:
            complexity_analysis: Complexity analysis results
            dependency_analysis: Dependency analysis results
            
        Returns:
            Quality score from 0-100
        """
        score = 100.0

        # Penalize high complexity
        high_complexity_ratio = complexity_analysis.get("high_complexity_files", 0) / max(1, complexity_analysis.get("total_files", 1))
        score -= high_complexity_ratio * 30

        # Penalize dependency cycles
        cycle_count = len(dependency_analysis.get("dependency_cycles", []))
        score -= min(cycle_count * 10, 30)

        # Penalize high coupling
        avg_deps = dependency_analysis.get("average_dependencies_per_file", 0)
        if avg_deps > 5:
            score -= min((avg_deps - 5) * 5, 20)

        return max(0, score)

    def _generate_recommendations(self, complexity_analysis: Dict[str, Any], dependency_analysis: Dict[str, Any], coupling_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for code quality improvement.
        
        Args:
            complexity_analysis: Complexity analysis results
            dependency_analysis: Dependency analysis results
            coupling_analysis: Coupling analysis results
            
        Returns:
            List of recommendations
        """
        recommendations = []

        # Complexity recommendations
        if complexity_analysis.get("high_complexity_files", 0) > 0:
            recommendations.append(
                f"Consider refactoring {complexity_analysis['high_complexity_files']} high-complexity files"
            )

        # Dependency recommendations
        if dependency_analysis.get("dependency_cycles"):
            recommendations.append(
                f"Break {len(dependency_analysis['dependency_cycles'])} dependency cycles to improve maintainability"
            )

        # Coupling recommendations
        high_coupling_files = len([f for f in coupling_analysis.get("efferent_coupling", {}).values() if f > 10])
        if high_coupling_files > 0:
            recommendations.append(
                f"Reduce coupling in {high_coupling_files} highly coupled files"
            )

        if not recommendations:
            recommendations.append("Code quality metrics look good! Keep up the good work.")

        return recommendations

    def display_repository_overview(self, overview: Dict[str, Any]) -> None:
        """Display repository overview in formatted tables.
        
        Args:
            overview: Repository overview data
        """
        repo_stats = overview["repository_stats"]
        
        # Main statistics table
        stats_table = Table(title="Repository Statistics", show_header=True, header_style="bold cyan")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right", style="green")
        
        stats_table.add_row("Total Files", str(repo_stats["total_files"]))
        stats_table.add_row("Total Lines", f"{repo_stats['total_lines']:,}")
        stats_table.add_row("Modules", str(repo_stats["modules"]))
        
        self.console.print(stats_table)

        # Language breakdown
        if repo_stats.get("languages"):
            lang_table = Table(title="Languages", show_header=True, header_style="bold yellow")
            lang_table.add_column("Language", style="yellow")
            lang_table.add_column("Files", justify="right", style="green")
            
            for language, count in sorted(repo_stats["languages"].items(), key=lambda x: x[1], reverse=True):
                lang_table.add_row(language, str(count))
            
            self.console.print(lang_table)

    def display_complexity_analysis(self, complexity_analysis: Dict[str, Any]) -> None:
        """Display complexity analysis in formatted tables.
        
        Args:
            complexity_analysis: Complexity analysis data
        """
        # Complexity overview
        complexity_table = Table(title="Complexity Analysis", show_header=True, header_style="bold red")
        complexity_table.add_column("Metric", style="red")
        complexity_table.add_column("Value", justify="right", style="white")
        
        complexity_table.add_row("Average Complexity", f"{complexity_analysis.get('average_complexity', 0):.2f}")
        complexity_table.add_row("Max Complexity", str(complexity_analysis.get('max_complexity', 0)))
        complexity_table.add_row("High Complexity Files", str(complexity_analysis.get('high_complexity_files', 0)))
        
        self.console.print(complexity_table)

        # Complexity distribution
        distribution = complexity_analysis.get("complexity_distribution", {})
        if distribution:
            dist_table = Table(title="Complexity Distribution", show_header=True, header_style="bold orange")
            dist_table.add_column("Range", style="orange")
            dist_table.add_column("Files", justify="right", style="white")
            
            for range_name, count in distribution.items():
                dist_table.add_row(range_name, str(count))
            
            self.console.print(dist_table)

    def display_dependency_cycles(self, cycles: List[List[str]]) -> None:
        """Display dependency cycles.
        
        Args:
            cycles: List of dependency cycles
        """
        if not cycles:
            self.console.print(Panel(
                f"[{Colors.SUCCESS}]✅ No dependency cycles detected![/]",
                title="Dependency Cycles",
                expand=False
            ))
            return

        self.console.print(f"\n[{Colors.WARNING_BOLD}]⚠️  Found {len(cycles)} dependency cycle(s):[/]")
        
        for i, cycle in enumerate(cycles[:5], 1):  # Show first 5 cycles
            cycle_text = " → ".join(os.path.basename(f) for f in cycle)
            self.console.print(f"[{Colors.WARNING}]{i}. {cycle_text}[/]")
        
        if len(cycles) > 5:
            self.console.print(f"[{Colors.DIM}]... and {len(cycles) - 5} more cycles[/]") 