"""docfusion CLI: ``docfusion triage|convert|batch|audit|preflight|bench``."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from docfusion.config import PipelineConfig
from docfusion.licenses import audit
from docfusion.triage.heuristics import triage_pdf

# Every write is explicitly UTF-8. Extracted documents are full of em-dashes,
# Greek letters and LaTeX; on Windows the default locale codec (cp1252) raises
# UnicodeEncodeError on all of them.
ENCODING = "utf-8"


def _add_pipeline_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vlm-base-url", default=None, help="OpenAI-compatible endpoint serving olmOCR.")
    parser.add_argument("--vlm-model", default=None, help="Model name to request from that endpoint.")
    parser.add_argument("--no-docling", action="store_true", help="Use the raw text layer for Tier 1.")
    parser.add_argument("--tier1-only", action="store_true",
                        help="Never call the VLM; report would-be escalations only.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Concurrent in-flight Tier-2 page requests within one document.")


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    if getattr(args, "vlm_base_url", None):
        cfg.vlm.base_url = args.vlm_base_url
    if getattr(args, "vlm_model", None):
        cfg.vlm.model = args.vlm_model
    if getattr(args, "no_docling", False):
        cfg.use_docling_tier1 = False
    if getattr(args, "tier1_only", False):
        cfg.tier2_enabled = False
    if getattr(args, "workers", None):
        cfg.max_tier2_workers = args.workers
    return cfg


def _print_specialist_bom() -> None:
    """Specialists declare a licence at registration; show it here.

    They are optional and lazily loaded, so an uninstalled one is listed rather
    than hidden — the point is that you can see what *would* enter the BOM if
    you installed it, not only what already has.
    """
    try:
        import docfusion.specialists.charts  # noqa: F401
        import docfusion.specialists.formulas  # noqa: F401
        from docfusion.specialists.base import registry_bom
    except ImportError:
        return

    rows = registry_bom()
    if not rows:
        return
    print("\n  optional per-domain specialists:")
    for row in rows:
        state = "installed" if row["installed"] else "not installed"
        kinds = ",".join(row["kinds"])
        print(f"    [{row['licence']:>10}] {row['name']:<14} {kinds:<14} "
              f"{row['origin']:<24} {state}")


def _cmd_audit() -> int:
    res = audit()
    for c in res.bill_of_materials:
        print(f"  [{c.license_class.value:>10}] {c.name:<22} {c.kind:<8} {c.license} ({c.developer})")
    _print_specialist_bom()
    if res.ok:
        print("\nAUDIT PASSED: all runtime components are enterprise-permissive.")
        return 0
    print("\nAUDIT FAILED:")
    for v in res.violations:
        print(f"  - {v}")
    return 1


def _cmd_preflight(args: argparse.Namespace) -> int:
    from docfusion.hardware import plan_serving

    plan = plan_serving(requested_model=args.vlm_model)
    for gpu in plan.gpus:
        fp8 = "fp8-capable" if gpu.supports_fp8 else "NO fp8 (pre-Ada)"
        print(f"  GPU {gpu.index}: {gpu.name}  {gpu.memory_total_gb:.0f} GB  "
              f"cc {gpu.compute_capability}  [{fp8}]")
    if not plan.gpus:
        print("  no NVIDIA GPU detected")

    print(f"\n  model            {plan.model}")
    print(f"  quantization     {plan.quantization}")
    print(f"  max-model-len    {plan.max_model_len}")
    print(f"  gpu-mem-util     {plan.gpu_memory_utilization}")
    print(f"\n  vllm serve {' '.join(plan.as_vllm_args())}")

    for w in plan.warnings:
        print(f"\n  WARNING: {w}")
    for e in plan.errors:
        print(f"\n  ERROR: {e}")
    return 0 if plan.ok else 1


def _cmd_triage(args: argparse.Namespace) -> int:
    decisions = triage_pdf(args.pdf)
    out = [
        {
            "page": d.profile.index,
            "route": d.route.value,
            "reasons": d.reasons,
            "chars": d.profile.char_count,
            "math_density": round(d.profile.math_density, 4),
            "image_area": round(d.profile.image_area_ratio, 3),
            "paths": d.profile.path_object_count,
        }
        for d in decisions
    ]
    print(json.dumps(out, indent=2))
    n_vlm = sum(1 for d in decisions if d.route.value == "vlm")
    print(f"# {len(decisions)} pages: {len(decisions) - n_vlm} fast, {n_vlm} vlm", file=sys.stderr)
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    from docfusion.pipeline import DocFusionPipeline

    result = DocFusionPipeline(_build_config(args)).convert(args.pdf)
    if args.output == "-":
        print(result.markdown)
    else:
        Path(args.output).write_text(result.markdown, encoding=ENCODING)
    print(
        f"# tier2 pages: {result.tier2_pages} ({result.tier2_fraction:.0%}); "
        f"degraded: {result.degraded_pages}",
        file=sys.stderr,
    )
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    from docfusion.pipeline import DocFusionPipeline

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from docfusion.io import IMAGE_SUFFIXES

    # Scans arrive as images as often as PDFs; batch should not silently skip them.
    suffixes = {".pdf", *IMAGE_SUFFIXES}
    pdfs = sorted(
        p for p in in_dir.glob("**/*") if p.is_file() and p.suffix.lower() in suffixes
    )
    if not pdfs:
        print(f"no PDFs or images found under {in_dir}", file=sys.stderr)
        return 0

    pipe = DocFusionPipeline(_build_config(args))
    lock = threading.Lock()
    failures = 0

    def convert_one(pdf: Path) -> None:
        nonlocal failures
        dest = out_dir / pdf.relative_to(in_dir).with_suffix(".md")
        if args.skip_existing and dest.exists():
            with lock:
                print(f"skip {pdf.name} (exists)", file=sys.stderr)
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = pipe.convert(pdf)
            dest.write_text(result.markdown, encoding=ENCODING)
            line = (f"ok   {pdf.name}: {len(result.decisions)} pages, "
                    f"tier2={len(result.tier2_pages)} ({result.tier2_fraction:.0%})"
                    + (f", degraded={result.degraded_pages}" if result.degraded_pages else ""))
        except Exception as exc:  # keep the batch moving; report at the end
            with lock:
                failures += 1
                print(f"FAIL {pdf.name}: {exc}", file=sys.stderr)
            return
        with lock:
            print(line, file=sys.stderr)

    # Document-level concurrency matters independently of --workers: that one
    # parallelises pages *within* a document, which does nothing for a corpus of
    # one-page scans. Total in-flight requests is doc_workers x workers.
    doc_workers = max(1, args.doc_workers)
    if doc_workers > 1:
        with ThreadPoolExecutor(max_workers=doc_workers) as pool:
            list(pool.map(convert_one, pdfs))
    else:
        for pdf in pdfs:
            convert_one(pdf)

    print(f"# done: {len(pdfs) - failures}/{len(pdfs)} converted", file=sys.stderr)
    return 1 if failures else 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from docfusion.bench import BenchPaths, prepare_candidate, score_candidate
    from docfusion.bench.harness import write_report_json
    from docfusion.pipeline import DocFusionPipeline

    paths = BenchPaths(Path(args.bench_dir))
    candidate = None

    if args.stage in ("prepare", "all"):
        pipe = DocFusionPipeline(_build_config(args))

        def progress(i: int, total: int, pdf: Path) -> None:
            if i % 25 == 0 or i == total:
                print(f"  [{i}/{total}] {pdf.name}", file=sys.stderr)

        candidate = prepare_candidate(
            paths,
            convert=pipe.convert,
            name=args.name,
            categories=args.categories,
            limit=args.limit,
            skip_existing=not args.force,
            workers=args.doc_workers,
            on_progress=progress,
        )
        print(
            f"# prepared {candidate.pdfs_converted} PDFs "
            f"({candidate.pdfs_failed} failed) into {candidate.directory}\n"
            f"# escalated {candidate.pages_escalated}/{candidate.pages_total} pages "
            f"({candidate.escalation_rate:.1%})",
            file=sys.stderr,
        )
        for name, why in candidate.failures[:10]:
            print(f"  FAIL {name}: {why}", file=sys.stderr)

    if args.stage in ("score", "all"):
        report = score_candidate(
            paths, name=args.name, python_exe=args.python_exe, olmocr_repo=args.olmocr_repo
        )
        print(report.raw_stdout)
        if args.json_out:
            write_report_json(report, candidate, Path(args.json_out))
            print(f"# wrote {args.json_out}", file=sys.stderr)
        if not report.ok:
            print(
                "# scorer did not produce an overall score — see output above",
                file=sys.stderr,
            )
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docfusion")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_triage = sub.add_parser("triage", help="Profile pages and show FAST/VLM routing (no models needed).")
    p_triage.add_argument("pdf")

    p_convert = sub.add_parser("convert", help="Run the dual-tier pipeline to Markdown.")
    p_convert.add_argument("pdf")
    p_convert.add_argument("-o", "--output", default="-")
    _add_pipeline_flags(p_convert)

    sub.add_parser("audit", help="Run the license compliance audit on the runtime BOM.")

    p_pre = sub.add_parser("preflight", help="Check GPU capability and print a vLLM serving plan.")
    p_pre.add_argument("--vlm-model", default=None)

    p_batch = sub.add_parser("batch", help="Convert every PDF in a directory to Markdown.")
    p_batch.add_argument("input_dir")
    p_batch.add_argument("output_dir")
    p_batch.add_argument("--skip-existing", action="store_true")
    p_batch.add_argument("--doc-workers", type=int, default=1,
                         help="Convert this many documents concurrently. Raise it for corpora of "
                              "single-page files, where --workers cannot help. Total in-flight "
                              "requests is --doc-workers x --workers.")
    _add_pipeline_flags(p_batch)

    p_bench = sub.add_parser("bench", help="Run olmOCR-Bench against this pipeline.")
    p_bench.add_argument("bench_dir", help="Path to a downloaded allenai/olmOCR-bench snapshot.")
    p_bench.add_argument("--stage", choices=["prepare", "score", "all"], default="all")
    p_bench.add_argument("--name", default="docfusion", help="Candidate directory name.")
    p_bench.add_argument("--categories", nargs="*", default=None,
                         help="Restrict to these pdfs/<category> subdirectories.")
    p_bench.add_argument("--limit", type=int, default=None, help="Only the first N PDFs (smoke runs).")
    p_bench.add_argument("--force", action="store_true", help="Re-convert PDFs that already have output.")
    p_bench.add_argument("--python-exe", default=None,
                         help="Interpreter that has olmOCR's bench extras installed.")
    p_bench.add_argument("--olmocr-repo", default=None, help="Override the vendored olmOCR submodule path.")
    p_bench.add_argument("--json-out", default=None, help="Write a machine-readable report here.")
    p_bench.add_argument("--doc-workers", type=int, default=8,
                         help="Convert this many bench PDFs concurrently. Bench PDFs are single "
                              "pages, so this is the only knob that keeps the GPU busy.")
    _add_pipeline_flags(p_bench)

    args = parser.parse_args(argv)

    if args.cmd == "audit":
        return _cmd_audit()
    if args.cmd == "preflight":
        return _cmd_preflight(args)
    if args.cmd == "triage":
        return _cmd_triage(args)
    if args.cmd == "convert":
        return _cmd_convert(args)
    if args.cmd == "batch":
        return _cmd_batch(args)
    if args.cmd == "bench":
        return _cmd_bench(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
