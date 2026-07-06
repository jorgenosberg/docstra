"""Impact stage: compute which files need documentation regeneration."""

from __future__ import annotations

from typing import Iterable, Sequence, Set

from docstra.core.indexing.code_index import CodebaseIndex


def graph_neighbors(index: CodebaseIndex, file_id: str) -> Set[str]:
    """Return files connected to ``file_id`` by an import edge, either direction."""
    refs = index.get_file_cross_references(file_id)
    return set(refs.get("imports", [])) | set(refs.get("imported_by", []))


def compute_impacted_file_ids(
    changed_ids: Iterable[str], indexes: Sequence[CodebaseIndex]
) -> Set[str]:
    """Return the changed files plus their graph neighbors in every index.

    A file page renders both its imports and its importers, so a change to
    one file invalidates the pages of its direct neighbors in both
    directions. Passing the pre-change and post-change indexes covers edges
    that the change itself added or removed.
    """
    changed = set(changed_ids)
    impacted = set(changed)
    for index in indexes:
        for file_id in changed:
            impacted |= graph_neighbors(index, file_id)
    return impacted
