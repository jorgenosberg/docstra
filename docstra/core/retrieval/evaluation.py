"""Small retrieval evaluation harness for Docstra's built-in codebase checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class RetrievalEvalCase:
    """A question and the source files that should appear in retrieval results."""

    question: str
    expected_files: List[str]


@dataclass(frozen=True)
class RetrievalEvalResult:
    """The outcome for one retrieval eval case."""

    case: RetrievalEvalCase
    retrieved_files: List[str]
    matched_file: Optional[str]
    rank: Optional[int]

    @property
    def passed(self) -> bool:
        """Return whether the case found an expected file."""
        return self.matched_file is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the case result for JSON output."""
        return {
            "question": self.case.question,
            "expected_files": self.case.expected_files,
            "retrieved_files": self.retrieved_files,
            "matched_file": self.matched_file,
            "rank": self.rank,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class RetrievalEvalSummary:
    """Aggregate retrieval eval results."""

    results: List[RetrievalEvalResult]
    top_k: int

    @property
    def total(self) -> int:
        """Return the number of evaluated cases."""
        return len(self.results)

    @property
    def passed_count(self) -> int:
        """Return the number of cases with an expected file in top-k."""
        return sum(1 for result in self.results if result.passed)

    @property
    def recall_at_k(self) -> float:
        """Return recall@k across all eval cases."""
        if self.total == 0:
            return 0.0
        return self.passed_count / self.total

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the summary for JSON output."""
        return {
            "top_k": self.top_k,
            "total": self.total,
            "passed": self.passed_count,
            "recall_at_k": self.recall_at_k,
            "results": [result.to_dict() for result in self.results],
        }


DEFAULT_RETRIEVAL_EVAL_CASES: List[RetrievalEvalCase] = [
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


RetrieverFn = Callable[[str, int], Sequence[Dict[str, Any]]]


def normalize_source_path(
    source_path: str, codebase_path: Optional[Path] = None
) -> str:
    """Normalize a retrieved source path for stable comparisons."""
    if not source_path:
        return ""

    path = Path(str(source_path)).expanduser()
    if path.is_absolute() and codebase_path:
        try:
            path = path.resolve().relative_to(codebase_path.resolve())
        except ValueError:
            path = path.resolve()

    normalized = path.as_posix()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def source_path_matches(
    retrieved_file: str,
    expected_file: str,
    codebase_path: Optional[Path] = None,
) -> bool:
    """Return whether a retrieved path matches an expected source file."""
    retrieved = normalize_source_path(retrieved_file, codebase_path)
    expected = normalize_source_path(expected_file, codebase_path)
    return retrieved == expected or retrieved.endswith(f"/{expected}")


def collect_retrieved_files(
    retrieval_results: Iterable[Dict[str, Any]],
    top_k: int,
    codebase_path: Optional[Path] = None,
) -> List[str]:
    """Extract ordered unique source files from retrieval result metadata."""
    retrieved_files: List[str] = []
    seen_files: set[str] = set()

    for result in retrieval_results:
        metadata = result.get("metadata") or {}
        source_path = (
            metadata.get("document_id")
            or metadata.get("filepath")
            or metadata.get("file_path")
            or metadata.get("path")
            or result.get("document_id")
            or result.get("filepath")
        )
        normalized_path = normalize_source_path(str(source_path or ""), codebase_path)
        if not normalized_path or normalized_path in seen_files:
            continue

        retrieved_files.append(normalized_path)
        seen_files.add(normalized_path)
        if len(retrieved_files) >= top_k:
            break

    return retrieved_files


def evaluate_retrieval_case(
    case: RetrievalEvalCase,
    retrieval_results: Sequence[Dict[str, Any]],
    top_k: int,
    codebase_path: Optional[Path] = None,
) -> RetrievalEvalResult:
    """Evaluate one case against already-fetched retrieval results."""
    retrieved_files = collect_retrieved_files(retrieval_results, top_k, codebase_path)

    for rank, retrieved_file in enumerate(retrieved_files, start=1):
        for expected_file in case.expected_files:
            if source_path_matches(retrieved_file, expected_file, codebase_path):
                return RetrievalEvalResult(
                    case=case,
                    retrieved_files=retrieved_files,
                    matched_file=retrieved_file,
                    rank=rank,
                )

    return RetrievalEvalResult(
        case=case,
        retrieved_files=retrieved_files,
        matched_file=None,
        rank=None,
    )


def evaluate_retrieval_cases(
    cases: Sequence[RetrievalEvalCase],
    retrieve: RetrieverFn,
    top_k: int = 10,
    candidate_k: Optional[int] = None,
    codebase_path: Optional[Path] = None,
) -> RetrievalEvalSummary:
    """Run a suite of retrieval eval cases using the supplied retriever."""
    candidate_limit = candidate_k or top_k
    results = [
        evaluate_retrieval_case(
            case=case,
            retrieval_results=retrieve(case.question, candidate_limit),
            top_k=top_k,
            codebase_path=codebase_path,
        )
        for case in cases
    ]
    return RetrievalEvalSummary(results=results, top_k=top_k)
