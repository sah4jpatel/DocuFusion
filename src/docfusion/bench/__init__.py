"""olmOCR-Bench integration.

Scores are produced by olmOCR's *own* scorer (``third_party/olmocr``), never a
reimplementation, so numbers are directly comparable to the published table.
This package only supplies the half upstream does not: turning a docfusion run
into the candidate layout the scorer expects.
"""

from docfusion.bench.harness import (
    BenchCandidate,
    BenchPaths,
    CategoryScore,
    ScoreReport,
    bench_pdfs,
    build_subset,
    candidate_relpath,
    prepare_candidate,
    score_candidate,
)

__all__ = [
    "BenchCandidate",
    "BenchPaths",
    "CategoryScore",
    "ScoreReport",
    "bench_pdfs",
    "build_subset",
    "candidate_relpath",
    "prepare_candidate",
    "score_candidate",
]
