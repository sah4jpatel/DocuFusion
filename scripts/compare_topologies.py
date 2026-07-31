"""Measure what triage actually costs, on real olmOCR-Bench tests.

Three candidates over the same PDFs:

``vlm_only``    every page to olmOCR-2 — the accuracy and cost ceiling
``hybrid``      default triage — what docfusion ships
``tier1_only``  no VLM at all — the deterministic floor

The gap between ``vlm_only`` and ``hybrid`` is the price of the GPU time triage
saves. Without this number the routing thresholds are just assertions.

    python scripts/compare_topologies.py .bench_data --categories tables multi_column
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docfusion.bench.harness import (  # noqa: E402
    BenchPaths,
    build_subset,
    prepare_candidate,
    score_candidate,
)
from docfusion.config import PipelineConfig  # noqa: E402
from docfusion.pipeline import DocFusionPipeline  # noqa: E402

TOPOLOGIES = {
    "vlm_only":   dict(force_tier2_all=True,  tier2_enabled=True),
    "hybrid":     dict(force_tier2_all=False, tier2_enabled=True),
    "tier1_only": dict(force_tier2_all=False, tier2_enabled=False),
}


def build_config(name: str, base_url: str, model: str, workers: int, docling: bool) -> PipelineConfig:
    cfg = PipelineConfig(**TOPOLOGIES[name])
    cfg.vlm.base_url = base_url
    cfg.vlm.model = model
    cfg.max_tier2_workers = workers
    cfg.use_docling_tier1 = docling
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_dir")
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--subset-dir", default=".bench_subset")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="allenai/olmOCR-2-7B-1025")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-docling", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="Run a subset of topologies.")
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()

    source = BenchPaths(Path(args.bench_dir))
    subset_root = Path(args.subset_dir)
    if not (subset_root / "bench_data" / "pdfs").is_dir():
        print(f"building subset {args.categories} -> {subset_root}", flush=True)
        build_subset(source, args.categories, subset_root)
    paths = BenchPaths(subset_root)
    total_pdfs = len(list(paths.pdf_dir.rglob("*.pdf")))
    print(f"subset: {total_pdfs} PDFs across {args.categories}\n", flush=True)

    results: dict[str, dict] = {}
    names = args.only or list(TOPOLOGIES)

    for name in names:
        cfg = build_config(name, args.base_url, args.model, args.workers, not args.no_docling)
        pipe = DocFusionPipeline(cfg)

        def progress(i: int, total: int, pdf: Path, _name=name) -> None:
            if i % 20 == 0 or i == total:
                print(f"  [{_name}] {i}/{total}", flush=True)

        print(f"=== {name}: converting ===", flush=True)
        started = time.perf_counter()
        candidate = prepare_candidate(paths, convert=pipe.convert, name=name,
                                      workers=args.workers, on_progress=progress)
        elapsed = time.perf_counter() - started
        print(f"=== {name}: {elapsed:.0f}s, {candidate.pdfs_failed} failures, "
              f"escalated {candidate.pages_escalated}/{candidate.pages_total}", flush=True)

        print(f"=== {name}: scoring ===", flush=True)
        report = score_candidate(paths, name=name)
        print(report.raw_stdout[-3000:], flush=True)

        results[name] = {
            "overall": report.overall,
            "ci_low": report.ci_low,
            "ci_high": report.ci_high,
            "categories": {c.name: c.score for c in report.categories},
            "convert_seconds": round(elapsed, 1),
            "pages_total": candidate.pages_total,
            "pages_escalated": candidate.pages_escalated,
            "escalation_rate": round(candidate.escalation_rate, 4),
            "pdfs_failed": candidate.pdfs_failed,
        }
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n================ SUMMARY ================")
    print(f"{'topology':<12}{'overall':>9}{'escalated':>11}{'convert s':>11}")
    for name, data in results.items():
        overall = f"{data['overall']:.1f}" if data["overall"] is not None else "n/a"
        print(f"{name:<12}{overall:>9}{data['escalation_rate']:>10.1%}{data['convert_seconds']:>11.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
