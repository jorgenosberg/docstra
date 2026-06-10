from __future__ import annotations

import json
import sys
from importlib import util
from pathlib import Path

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
