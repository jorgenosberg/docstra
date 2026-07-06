"""Documentation pipeline stages.

Each module is a pure, independently testable stage:
analyze -> plan -> (generate, in the generator) -> crosslink -> render.
impact computes which files need regeneration after a change.
"""

from docstra.core.documentation.pipeline.analyze import (
    CodebaseAnalysis,
    analyze_codebase,
    module_category,
)
from docstra.core.documentation.pipeline.crosslink import (
    render_cross_references_section,
)
from docstra.core.documentation.pipeline.impact import compute_impacted_file_ids
from docstra.core.documentation.pipeline.llms_txt import write_llms_txt
from docstra.core.documentation.pipeline.plan import (
    DocPlan,
    PlannedPage,
    doc_relative_path,
    file_doc_path,
    plan_documentation,
)

__all__ = [
    "CodebaseAnalysis",
    "DocPlan",
    "PlannedPage",
    "analyze_codebase",
    "compute_impacted_file_ids",
    "doc_relative_path",
    "file_doc_path",
    "module_category",
    "plan_documentation",
    "render_cross_references_section",
    "write_llms_txt",
]
