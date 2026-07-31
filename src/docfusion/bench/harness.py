"""Run docfusion against olmOCR-Bench and score it with upstream's scorer.

olmOCR-Bench checks machine-verifiable *facts* about a page — "this sentence
appears", "this header does not", "this equation renders equivalently", "cell
(2,3) is to the right of cell (2,2)" — instead of edit distance against a
reference. Two correct parses can differ textually, so pass/fail facts are the
honest metric.

Layout the scorer expects::

    bench_data/
      *.jsonl                 # the tests, one file per category
      pdfs/<category>/*.pdf   # single-page source PDFs
      <candidate>/            # one directory per system under test
        <pdf_stem>_pg<page>_repeat<n>.md

Only that last directory is ours to produce. Scoring is delegated to
``python -m olmocr.bench.benchmark`` so results stay comparable to the
published table, and so upstream fixes to equation comparison or table parsing
land here for free when the submodule is bumped.

The scorer runs out-of-process because it needs a heavier dependency set than
docfusion's core (Playwright + Chromium for KaTeX equation comparison,
rapidfuzz, fuzzysearch). ``python_exe`` selects the interpreter that has them —
typically a Linux venv, since Playwright and vLLM are Linux-first.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CANDIDATE = "docfusion"
_REPEAT_RE = re.compile(r"_pg(\d+)_repeat(\d+)\.md$")


@dataclass(frozen=True)
class BenchPaths:
    """Locations inside a downloaded ``allenai/olmOCR-bench`` snapshot."""

    root: Path

    @property
    def data_dir(self) -> Path:
        """The directory holding the ``.jsonl`` files and ``pdfs/``.

        The HF snapshot nests these one level down in ``bench_data/``; accept
        either the snapshot root or the data directory itself.
        """
        nested = self.root / "bench_data"
        return nested if (nested / "pdfs").is_dir() else self.root

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    def candidate_dir(self, name: str = DEFAULT_CANDIDATE) -> Path:
        return self.data_dir / name

    def jsonl_files(self) -> list[Path]:
        return sorted(self.data_dir.glob("*.jsonl"))

    def validate(self) -> None:
        if not self.pdf_dir.is_dir():
            raise FileNotFoundError(
                f"{self.pdf_dir} not found. Download the bench first:\n"
                f"  hf download allenai/olmOCR-bench --repo-type dataset --local-dir {self.root}"
            )
        if not self.jsonl_files():
            raise FileNotFoundError(f"no .jsonl test files under {self.data_dir}")


@dataclass
class BenchCandidate:
    """A prepared candidate directory plus what happened while producing it."""

    name: str
    directory: Path
    pdfs_converted: int = 0
    pdfs_failed: int = 0
    pages_escalated: int = 0
    pages_total: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def escalation_rate(self) -> float:
        return self.pages_escalated / self.pages_total if self.pages_total else 0.0


@dataclass
class CategoryScore:
    name: str
    score: float
    tests: int = 0


@dataclass
class ScoreReport:
    candidate: str
    overall: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    categories: list[CategoryScore] = field(default_factory=list)
    raw_stdout: str = ""
    returncode: int = 0

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.overall is not None


def bench_pdfs(paths: BenchPaths, categories: Iterable[str] | None = None) -> list[Path]:
    """Every bench PDF, optionally restricted to some category subdirectories."""
    paths.validate()
    wanted = set(categories) if categories else None
    pdfs: list[Path] = []
    for pdf in sorted(paths.pdf_dir.rglob("*.pdf")):
        category = pdf.parent.name
        if wanted is None or category in wanted:
            pdfs.append(pdf)
    return pdfs


def build_subset(paths: BenchPaths, categories: Iterable[str], dest: Path) -> BenchPaths:
    """Materialise a smaller, self-consistent bench containing whole categories.

    The scorer globs every ``*.jsonl`` under ``--dir`` and treats a PDF with no
    candidate output as an error that zeroes the run, so you cannot benchmark a
    slice by simply converting fewer files. Copying whole categories — PDFs plus
    the test rows that reference them — keeps each per-category score complete
    and therefore comparable to the published table.

    Category directory names and JSONL names do not correspond one-to-one
    (``pdfs/tables/`` is tested by ``table_tests.jsonl``), so rows are filtered
    by the category embedded in each test's ``pdf`` field.
    """
    paths.validate()
    wanted = set(categories)
    dest = Path(dest)
    data = dest / "bench_data"
    (data / "pdfs").mkdir(parents=True, exist_ok=True)

    kept_pdfs: set[str] = set()
    for category in sorted(wanted):
        source = paths.pdf_dir / category
        if not source.is_dir():
            raise FileNotFoundError(f"no such bench category: {source}")
        target = data / "pdfs" / category
        target.mkdir(parents=True, exist_ok=True)
        for pdf in sorted(source.glob("*.pdf")):
            shutil.copyfile(pdf, target / pdf.name)
            kept_pdfs.add(f"{category}/{pdf.name}")

    for jsonl in paths.jsonl_files():
        rows = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("pdf") in kept_pdfs:
                rows.append(line)
        if rows:
            (data / jsonl.name).write_text("\n".join(rows) + "\n", encoding="utf-8")

    return BenchPaths(dest)


def candidate_relpath(pdf_relative: Path | str, page: int = 1, repeat: int = 1) -> Path:
    """``tables/foo.pdf`` → ``tables/foo_pg1_repeat1.md``.

    The category directory is part of the key, not decoration: the scorer builds
    its regex from the test's ``pdf`` field (``tables/foo.pdf``) and matches it
    against each candidate file's path *relative to the candidate directory*.
    Writing the files flat makes every single test report "missing MD repeats"
    and the whole run scores 0.
    """
    relative = Path(pdf_relative)
    return relative.parent / f"{relative.stem}_pg{page}_repeat{repeat}.md"


def prepare_candidate(
    paths: BenchPaths,
    convert: Callable[[Path], object],
    name: str = DEFAULT_CANDIDATE,
    categories: Iterable[str] | None = None,
    repeats: int = 1,
    skip_existing: bool = True,
    limit: int | None = None,
    workers: int = 1,
    on_progress: Callable[[int, int, Path], None] | None = None,
) -> BenchCandidate:
    """Convert bench PDFs and write the candidate directory.

    ``convert`` takes a PDF path and returns either a ``DocumentResult`` or a
    plain string, so alternative topologies (Marker-driven, Docling-driven,
    Tier-1-only) can be benchmarked through the same harness.

    A PDF that fails conversion is recorded and skipped rather than aborting the
    run — but it still writes an empty file, because the scorer treats a
    *missing* file as a candidate error that zeroes the entire run, which would
    disguise one broken document as total failure.

    ``workers`` parallelises across *documents*. The pipeline's own
    ``max_tier2_workers`` only parallelises pages within one document, which
    yields exactly nothing on a corpus of single-page PDFs — the bench, and any
    scan-per-file archive. Without this the GPU sits idle between pages.
    """
    paths.validate()
    pdfs = bench_pdfs(paths, categories)
    if limit is not None:
        pdfs = pdfs[:limit]

    out_dir = paths.candidate_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = BenchCandidate(name=name, directory=out_dir)
    lock = threading.Lock()
    done = 0

    def handle(pdf: Path) -> None:
        nonlocal done
        relative = pdf.relative_to(paths.pdf_dir)
        targets = [out_dir / candidate_relpath(relative, 1, r) for r in range(1, repeats + 1)]

        if skip_existing and all(t.exists() for t in targets):
            with lock:
                candidate.pdfs_converted += 1
                done += 1
                if on_progress:
                    on_progress(done, len(pdfs), pdf)
            return

        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                result = convert(pdf)
                markdown = getattr(result, "markdown", result)
                decisions = getattr(result, "decisions", None)
                target.write_text(str(markdown or ""), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001 — one bad PDF must not end the run
                target.write_text("", encoding="utf-8")
                with lock:
                    candidate.pdfs_failed += 1
                    candidate.failures.append((pdf.name, f"{type(exc).__name__}: {exc}"))
            else:
                with lock:
                    candidate.pdfs_converted += 1
                    if decisions is not None:
                        candidate.pages_total += len(decisions)
                        candidate.pages_escalated += len(getattr(result, "tier2_pages", []))
        with lock:
            done += 1
            if on_progress:
                on_progress(done, len(pdfs), pdf)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(handle, pdfs))
    else:
        for pdf in pdfs:
            handle(pdf)

    return candidate


# ---------------------------------------------------------------------------
# Scoring (delegated to upstream)
# ---------------------------------------------------------------------------

# Upstream's final summary line, e.g.
#   docfusion            : Average Score: 53.4% ± 1.3% (average of per-JSONL scores)
_OVERALL_CI_RE = re.compile(
    r"Average\s+Score:\s*([\d.]+)\s*%\s*(?:±|\+/-)\s*([\d.]+)\s*%", re.IGNORECASE
)
_OVERALL_RE = re.compile(r"Average\s+Score:\s*([\d.]+)\s*%", re.IGNORECASE)
# Per-file lines, e.g. "        multi_column.jsonl            : 65.6% (580/884 tests)"
_CATEGORY_RE = re.compile(
    r"^\s*([A-Za-z_][\w.-]*)\.jsonl\s*:\s*([\d.]+)\s*%(?:\s*\((\d+)/(\d+)\s*tests\))?",
    re.MULTILINE,
)


def parse_bench_output(stdout: str, candidate: str) -> ScoreReport:
    """Pull scores out of the scorer's console report.

    Upstream prints a human report rather than emitting JSON, so this is
    deliberately forgiving: an unparsed number leaves ``overall=None`` and the
    caller still has ``raw_stdout``. It never invents a score.
    """
    report = ScoreReport(candidate=candidate, raw_stdout=stdout)

    seen: set[str] = set()
    for name, value, _passed, total in _CATEGORY_RE.findall(stdout):
        if name in seen:
            continue
        seen.add(name)
        try:
            report.categories.append(
                CategoryScore(name=name, score=float(value), tests=int(total) if total else 0)
            )
        except ValueError:
            continue

    # Prefer the line that carries the confidence interval — the same run also
    # prints a bare per-candidate average that would otherwise win.
    ci = _OVERALL_CI_RE.search(stdout)
    if ci:
        try:
            report.overall = float(ci.group(1))
            margin = float(ci.group(2))
            report.ci_low = report.overall - margin
            report.ci_high = report.overall + margin
        except ValueError:
            pass
    else:
        overall = _OVERALL_RE.search(stdout)
        if overall:
            try:
                report.overall = float(overall.group(1))
            except ValueError:
                pass

    return report


def score_candidate(
    paths: BenchPaths,
    name: str = DEFAULT_CANDIDATE,
    olmocr_repo: Path | None = None,
    python_exe: str | None = None,
    timeout_s: int = 7200,
    extra_args: list[str] | None = None,
) -> ScoreReport:
    """Run upstream's scorer against a prepared candidate directory.

    ``olmocr_repo`` defaults to the vendored submodule, so scoring tracks
    whatever olmOCR revision the submodule is pinned to.
    """
    paths.validate()
    candidate_dir = paths.candidate_dir(name)
    if not candidate_dir.is_dir():
        raise FileNotFoundError(f"candidate directory {candidate_dir} does not exist; run prepare first")

    repo = Path(olmocr_repo) if olmocr_repo else Path(__file__).resolve().parents[3] / "third_party" / "olmocr"
    if not (repo / "olmocr" / "bench" / "benchmark.py").exists():
        raise FileNotFoundError(
            f"olmOCR bench scorer not found under {repo}. Initialise submodules:\n"
            f"  git submodule update --init --depth 1"
        )

    # The scorer runs with cwd set to the olmOCR repo so its package imports
    # resolve, which means a relative --dir would be interpreted there. Always
    # hand it an absolute path.
    cmd = [
        python_exe or sys.executable,
        "-m", "olmocr.bench.benchmark",
        "--dir", str(paths.data_dir.resolve()),
        "--candidate", name,
    ]
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.run(
        cmd, cwd=str(repo), capture_output=True, text=True, timeout=timeout_s, check=False
    )
    combined = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    report = parse_bench_output(combined, name)
    report.returncode = proc.returncode
    return report


def write_report_json(report: ScoreReport, candidate: BenchCandidate | None, path: Path) -> None:
    payload: dict[str, object] = {
        "candidate": report.candidate,
        "overall": report.overall,
        "ci_low": report.ci_low,
        "ci_high": report.ci_high,
        "categories": [{"name": c.name, "score": c.score} for c in report.categories],
        "returncode": report.returncode,
    }
    if candidate is not None:
        payload["conversion"] = {
            "pdfs_converted": candidate.pdfs_converted,
            "pdfs_failed": candidate.pdfs_failed,
            "pages_total": candidate.pages_total,
            "pages_escalated": candidate.pages_escalated,
            "escalation_rate": round(candidate.escalation_rate, 4),
            "failures": candidate.failures[:50],
        }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
