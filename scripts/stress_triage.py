"""Stress the triage router over a real corpus and report routing by category.

Run against olmOCR-Bench, whose directory names are a difficulty label:
``old_scans`` should escalate almost always, ``headers_footers`` rarely. That
turns an otherwise unfalsifiable heuristic into something measurable.

    python scripts/stress_triage.py .bench_data --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docfusion.config import TriageThresholds  # noqa: E402
from docfusion.triage.heuristics import Route, triage_pdf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_dir")
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.bench_dir)
    pdf_root = root / "bench_data" / "pdfs"
    if not pdf_root.is_dir():
        pdf_root = root / "pdfs"
    pdfs = sorted(pdf_root.rglob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print(f"no PDFs under {pdf_root}", file=sys.stderr)
        return 1

    thresholds = TriageThresholds()
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"fast": 0, "vlm": 0, "error": 0})
    reason_counts: dict[str, int] = defaultdict(int)
    per_page_times: list[float] = []
    errors: list[tuple[str, str]] = []
    total_pages = 0

    wall_start = time.perf_counter()
    for pdf in pdfs:
        category = pdf.parent.name
        started = time.perf_counter()
        try:
            decisions = triage_pdf(pdf, thresholds)
        except Exception as exc:  # noqa: BLE001 — this is the thing we are measuring
            by_category[category]["error"] += 1
            errors.append((str(pdf.relative_to(pdf_root)), f"{type(exc).__name__}: {exc}"))
            if len(errors) <= 3:
                traceback.print_exc(file=sys.stderr)
            continue
        elapsed = time.perf_counter() - started
        if decisions:
            per_page_times.append(elapsed / len(decisions))
        total_pages += len(decisions)
        for decision in decisions:
            by_category[category]["vlm" if decision.route is Route.VLM else "fast"] += 1
            for reason in decision.reasons:
                reason_counts[reason.split("(")[0].split("—")[0].strip()] += 1
    wall = time.perf_counter() - wall_start

    print(f"\n{len(pdfs)} PDFs / {total_pages} pages in {wall:.1f}s "
          f"({total_pages / wall:.0f} pages/sec)\n")
    print(f"{'category':<20} {'pages':>7} {'fast':>7} {'vlm':>7} {'escalated':>10}")
    print("-" * 56)
    grand_fast = grand_vlm = 0
    for category in sorted(by_category):
        counts = by_category[category]
        pages = counts["fast"] + counts["vlm"]
        grand_fast += counts["fast"]
        grand_vlm += counts["vlm"]
        rate = counts["vlm"] / pages if pages else 0.0
        print(f"{category:<20} {pages:>7} {counts['fast']:>7} {counts['vlm']:>7} {rate:>9.1%}"
              + (f"   ({counts['error']} errors)" if counts["error"] else ""))
    overall = grand_vlm / (grand_fast + grand_vlm) if (grand_fast + grand_vlm) else 0.0
    print("-" * 56)
    print(f"{'TOTAL':<20} {grand_fast + grand_vlm:>7} {grand_fast:>7} {grand_vlm:>7} {overall:>9.1%}")

    print("\ntrigger frequency:")
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>6}  {reason}")

    if per_page_times:
        print(f"\nper-page triage: median {statistics.median(per_page_times) * 1000:.2f} ms, "
              f"p95 {sorted(per_page_times)[int(len(per_page_times) * 0.95)] * 1000:.2f} ms")

    if errors:
        print(f"\n{len(errors)} PDFs raised:")
        for name, why in errors[:15]:
            print(f"  {name}: {why}")
    else:
        print("\nno PDFs raised during triage")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "pdfs": len(pdfs), "pages": total_pages, "wall_s": wall,
            "pages_per_sec": total_pages / wall if wall else 0,
            "by_category": {k: dict(v) for k, v in by_category.items()},
            "overall_escalation": overall,
            "reasons": dict(reason_counts),
            "errors": errors,
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
