from __future__ import annotations

import hashlib
import json
import struct
import sys
from importlib import util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docstra"
    / "core"
    / "retrieval"
    / "evaluation.py"
)
SPEC = util.spec_from_file_location("retrieval_evaluation", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
retrieval_evaluation = util.module_from_spec(SPEC)
sys.modules[SPEC.name] = retrieval_evaluation
SPEC.loader.exec_module(retrieval_evaluation)

RetrievalEvalCase = retrieval_evaluation.RetrievalEvalCase
collect_retrieved_files = retrieval_evaluation.collect_retrieved_files
evaluate_retrieval_case = retrieval_evaluation.evaluate_retrieval_case
evaluate_retrieval_cases = retrieval_evaluation.evaluate_retrieval_cases
normalize_source_path = retrieval_evaluation.normalize_source_path
source_path_matches = retrieval_evaluation.source_path_matches


def test_normalize_source_path_makes_absolute_paths_relative() -> None:
    codebase_path = Path("/repo/docstra")

    normalized = normalize_source_path(
        "/repo/docstra/docstra/core/cli.py", codebase_path
    )

    assert normalized == "docstra/core/cli.py"


def test_source_path_matches_absolute_and_relative_paths() -> None:
    codebase_path = Path("/repo/docstra")

    assert source_path_matches(
        "/repo/docstra/docstra/core/retrieval/chroma.py",
        "docstra/core/retrieval/chroma.py",
        codebase_path,
    )


def test_collect_retrieved_files_uses_ordered_unique_document_ids() -> None:
    retrieval_results = [
        {"metadata": {"document_id": "docstra/core/cli.py"}},
        {"metadata": {"document_id": "docstra/core/cli.py"}},
        {"metadata": {"document_id": "docstra/core/config/settings.py"}},
        {"metadata": {"document_id": "docstra/core/retrieval/chroma.py"}},
    ]

    retrieved_files = collect_retrieved_files(retrieval_results, top_k=2)

    assert retrieved_files == [
        "docstra/core/cli.py",
        "docstra/core/config/settings.py",
    ]


def test_evaluate_retrieval_case_reports_first_expected_file_rank() -> None:
    case = RetrievalEvalCase(
        question="Where is query implemented?",
        expected_files=["docstra/core/cli.py"],
    )
    retrieval_results = [
        {"metadata": {"document_id": "docstra/core/config/settings.py"}},
        {"metadata": {"document_id": "docstra/core/cli.py"}},
    ]

    result = evaluate_retrieval_case(case, retrieval_results, top_k=10)

    assert result.passed is True
    assert result.matched_file == "docstra/core/cli.py"
    assert result.rank == 2


def test_evaluate_retrieval_case_fails_without_expected_file() -> None:
    case = RetrievalEvalCase(
        question="Where is token budgeting implemented?",
        expected_files=["docstra/core/utils/token_counter.py"],
    )
    retrieval_results = [
        {"metadata": {"document_id": "docstra/core/cli.py"}},
        {"metadata": {"document_id": "docstra/core/config/settings.py"}},
    ]

    result = evaluate_retrieval_case(case, retrieval_results, top_k=10)

    assert result.passed is False
    assert result.matched_file is None
    assert result.rank is None


def test_evaluate_retrieval_cases_summarizes_recall_and_json() -> None:
    cases = [
        RetrievalEvalCase(
            question="Where is config loaded?",
            expected_files=["docstra/core/config/settings.py"],
        ),
        RetrievalEvalCase(
            question="Where is Chroma retrieval?",
            expected_files=["docstra/core/retrieval/chroma.py"],
        ),
    ]

    def retrieve(question: str, top_k: int):
        del top_k
        if "config" in question:
            return [{"metadata": {"document_id": "docstra/core/config/settings.py"}}]
        return [{"metadata": {"document_id": "docstra/core/cli.py"}}]

    summary = evaluate_retrieval_cases(cases, retrieve, top_k=5)
    payload = summary.to_dict()

    assert summary.total == 2
    assert summary.passed_count == 1
    assert summary.recall_at_k == 0.5
    assert payload["top_k"] == 5
    assert payload["passed"] == 1
    assert json.loads(json.dumps(payload)) == payload


def test_evaluate_retrieval_cases_can_overfetch_duplicate_chunks() -> None:
    case = RetrievalEvalCase(
        question="Where is the second relevant file?",
        expected_files=["b.py"],
    )
    requested_limits = []

    def retrieve(question: str, candidate_k: int):
        del question
        requested_limits.append(candidate_k)
        results = [
            {"metadata": {"document_id": "a.py"}},
            {"metadata": {"document_id": "a.py"}},
            {"metadata": {"document_id": "b.py"}},
        ]
        return results[:candidate_k]

    summary = evaluate_retrieval_cases(
        [case],
        retrieve,
        top_k=2,
        candidate_k=3,
    )

    assert requested_limits == [3]
    assert summary.results[0].passed is True
    assert summary.results[0].rank == 2
    assert summary.results[0].retrieved_files == ["a.py", "b.py"]


# ---------------------------------------------------------------------------
# Integration: Chroma-only vs. FusionRetriever side-by-side eval
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Subset of real source files that map to the eval queries in evaluation.py.
# Keep the list small so the test stays fast (no network, local embeddings only).
_CORPUS_FILES: List[Dict[str, Any]] = [
    {
        "path": "docstra/core/config/settings.py",
        "abs": _REPO_ROOT / "docstra/core/config/settings.py",
    },
    {
        "path": "docstra/core/retrieval/chroma.py",
        "abs": _REPO_ROOT / "docstra/core/retrieval/chroma.py",
    },
    {
        "path": "docstra/core/retrieval/context_aware.py",
        "abs": _REPO_ROOT / "docstra/core/retrieval/context_aware.py",
    },
    {
        "path": "docstra/core/ingestion/storage.py",
        "abs": _REPO_ROOT / "docstra/core/ingestion/storage.py",
    },
    {
        "path": "docstra/core/services/ingestion_service.py",
        "abs": _REPO_ROOT / "docstra/core/services/ingestion_service.py",
    },
    {
        "path": "docstra/core/indexing/code_index.py",
        "abs": _REPO_ROOT / "docstra/core/indexing/code_index.py",
    },
    {
        "path": "docstra/core/cli.py",
        "abs": _REPO_ROOT / "docstra/core/cli.py",
    },
    {
        "path": "docstra/core/utils/token_counter.py",
        "abs": _REPO_ROOT / "docstra/core/utils/token_counter.py",
    },
    {
        "path": "docstra/core/documentation/generator.py",
        "abs": _REPO_ROOT / "docstra/core/documentation/generator.py",
    },
]

_EVAL_CASES = [
    RetrievalEvalCase(
        question="How does Docstra load and save user configuration?",
        expected_files=["docstra/core/config/settings.py"],
    ),
    RetrievalEvalCase(
        question="Where are files ingested into ChromaDB with embeddings?",
        expected_files=[
            "docstra/core/services/ingestion_service.py",
            "docstra/core/ingestion/storage.py",
        ],
    ),
    RetrievalEvalCase(
        question="How does Chroma retrieval search document chunks?",
        expected_files=["docstra/core/retrieval/chroma.py"],
    ),
    RetrievalEvalCase(
        question="Where does Docstra classify query intent for context retrieval?",
        expected_files=["docstra/core/retrieval/context_aware.py"],
    ),
    RetrievalEvalCase(
        question="How are code symbols indexed for later search?",
        expected_files=["docstra/core/indexing/code_index.py"],
    ),
    RetrievalEvalCase(
        question="Where is the docstra query CLI command implemented?",
        expected_files=["docstra/core/cli.py"],
    ),
    RetrievalEvalCase(
        question="How is context token budget calculated and enforced?",
        expected_files=["docstra/core/utils/token_counter.py"],
    ),
    RetrievalEvalCase(
        question="Where does documentation generation assemble code context?",
        expected_files=["docstra/core/documentation/generator.py"],
    ),
]


class _DummyEmbedder:
    """Returns a deterministic 8-d embedding so Chroma is happy without a real model."""

    def generate_embedding(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vals = struct.unpack(">8I", digest[:32])
        return [v / 0xFFFFFFFF for v in vals]


def _split_into_chunks(content: str, chunk_size: int = 40) -> List[Dict[str, Any]]:
    """Split file content into fixed-size line chunks for the eval corpus."""
    lines = content.splitlines(keepends=True)
    chunks = []
    for start in range(0, len(lines), chunk_size):
        end = min(start + chunk_size, len(lines))
        chunk_text = "".join(lines[start:end])
        chunks.append(
            {
                "start_line": start + 1,
                "end_line": end,
                "content": chunk_text,
            }
        )
    return chunks


def test_chroma_vs_fusion_retrieval_eval(tmp_path: Path) -> None:
    """Build a small real corpus and compare Chroma-only vs. FusionRetriever recall."""
    import pytest

    # Skip if any corpus file is missing (e.g., running from a partial checkout).
    missing = [f["path"] for f in _CORPUS_FILES if not f["abs"].exists()]
    if missing:
        pytest.skip(f"corpus files not found: {missing}")

    from docstra.core.ingestion.fts_storage import FtsStorage
    from docstra.core.ingestion.storage import ChromaDBStorage
    from docstra.core.retrieval.chroma import ChromaRetriever
    from docstra.core.retrieval.fts import FtsRetriever
    from docstra.core.retrieval.fusion import FusionRetriever

    chroma_storage = ChromaDBStorage(persist_directory=str(tmp_path / "chroma"))
    fts_storage = FtsStorage(str(tmp_path / "index.db"))
    embedder = _DummyEmbedder()

    # Registry for the minimal in-memory CodebaseIndex substitute.
    # Maps file_id -> [(chunk_id, start_line, end_line)]
    chunks_by_file: Dict[str, List] = {}
    file_language: Dict[str, str] = {}

    for corpus_entry in _CORPUS_FILES:
        file_id: str = corpus_entry["path"]
        abs_path: Path = corpus_entry["abs"]
        content = abs_path.read_text(encoding="utf-8", errors="replace")
        file_chunks = _split_into_chunks(content)
        language = "python"

        chunk_ids = []
        chunk_contents = []
        chunk_metadatas = []
        chunk_embeddings = []
        fts_chunk_ids = []
        fts_start_lines = []
        fts_end_lines = []
        fts_contents = []

        file_chunk_tuples = []
        for fc in file_chunks:
            chunk_id = f"{file_id}#L{fc['start_line']}-L{fc['end_line']}"
            embedding = embedder.generate_embedding(fc["content"])

            chunk_ids.append(chunk_id)
            chunk_contents.append(fc["content"])
            chunk_metadatas.append(
                {
                    "document_id": file_id,
                    "start_line": fc["start_line"],
                    "end_line": fc["end_line"],
                    "language": language,
                    "chunk_id": chunk_id,
                }
            )
            chunk_embeddings.append(embedding)

            fts_chunk_ids.append(chunk_id)
            fts_start_lines.append(fc["start_line"])
            fts_end_lines.append(fc["end_line"])
            fts_contents.append(fc["content"])

            file_chunk_tuples.append((chunk_id, fc["start_line"], fc["end_line"]))

        chroma_storage.add_chunks(
            chunk_ids=chunk_ids,
            contents=chunk_contents,
            metadatas=chunk_metadatas,
            embeddings=chunk_embeddings,
        )
        fts_storage.add_chunks(
            chunk_ids=fts_chunk_ids,
            file_ids=[file_id] * len(fts_chunk_ids),
            languages=[language] * len(fts_chunk_ids),
            start_lines=fts_start_lines,
            end_lines=fts_end_lines,
            contents=fts_contents,
        )
        chunks_by_file[file_id] = file_chunk_tuples
        file_language[file_id] = language

    # Minimal CodebaseIndex substitute: only needs chunks_for_file + file_language.
    code_index = SimpleNamespace(
        chunks_for_file=lambda fid: chunks_by_file.get(fid, []),
        file_language=lambda fid: file_language.get(fid),
    )

    chroma_retriever = ChromaRetriever(chroma_storage, embedder)
    fts_retriever = FtsRetriever(fts_storage)

    fusion_retriever = FusionRetriever(
        dense=chroma_retriever,
        fts=fts_retriever,
        code_index=code_index,
    )

    top_k = 5

    def chroma_retrieve(question: str, n: int) -> List[Dict[str, Any]]:
        return chroma_retriever.retrieve_chunks(question, n_results=n)

    def fusion_retrieve(question: str, n: int) -> List[Dict[str, Any]]:
        return fusion_retriever.retrieve_chunks(question, n_results=n)

    chroma_summary = evaluate_retrieval_cases(_EVAL_CASES, chroma_retrieve, top_k=top_k)
    fusion_summary = evaluate_retrieval_cases(_EVAL_CASES, fusion_retrieve, top_k=top_k)

    # --- Print side-by-side table (visible with pytest -s) ---
    print(f"\n{'':=<72}")
    print(f"Retrieval eval  top_k={top_k}  corpus={len(_CORPUS_FILES)} files")
    print(f"{'':=<72}")
    header = f"{'Query':<50} {'Chroma':>7} {'Fusion':>7}"
    print(header)
    print(f"{'':-<72}")
    for cr, fr in zip(chroma_summary.results, fusion_summary.results):
        chroma_rank = str(cr.rank) if cr.rank else "-"
        fusion_rank = str(fr.rank) if fr.rank else "-"
        q = cr.case.question[:48]
        print(f"{q:<50} {chroma_rank:>7} {fusion_rank:>7}")
    print(f"{'':-<72}")
    print(
        f"{'recall@k':<50} {chroma_summary.recall_at_k:>7.2f} {fusion_summary.recall_at_k:>7.2f}"
    )
    print(
        f"{'passed':<50} {chroma_summary.passed_count:>7} {fusion_summary.passed_count:>7}"
    )
    print(f"{'':=<72}")

    # Sanity: scores must be valid floats, not errors.
    assert 0.0 <= chroma_summary.recall_at_k <= 1.0
    assert 0.0 <= fusion_summary.recall_at_k <= 1.0
