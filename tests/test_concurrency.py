"""Concurrency safety around PDFium.

PDFium keeps global state and is not thread-safe. Before
:mod:`docfusion.pdfium_lock` existed, converting documents in parallel killed
the process with ``munmap_chunk(): invalid pointer`` — a native abort with no
Python traceback, so it surfaced as "the batch job just died".

A native crash cannot be caught with ``pytest.raises``; if the lock regresses,
these tests take the whole worker down. That is the intended signal: a hard
failure in CI beats a segfault in a production batch run.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pypdfium2 as pdfium

from docfusion.engines.docling_engine import extract_text_layer
from docfusion.engines.olmocr_protocol import render_page_png
from docfusion.pdfium_lock import pdfium_guard
from docfusion.triage.heuristics import triage_pdf

THREADS = 8
ROUNDS = 6


class TestParallelPdfiumAccess:
    def test_parallel_triage_is_safe(self, mixed_pdf, simple_pdf, math_pdf):
        pdfs = [mixed_pdf, simple_pdf, math_pdf] * ROUNDS

        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            results = list(pool.map(triage_pdf, pdfs))

        assert len(results) == len(pdfs)
        assert all(decisions for decisions in results)

    def test_parallel_render_is_safe(self, simple_pdf, math_pdf):
        def render(path) -> int:
            with pdfium_guard():
                pdf = pdfium.PdfDocument(str(path))
                page = pdf[0]
            try:
                return len(render_page_png(page, target_longest_dim=512))
            finally:
                with pdfium_guard():
                    page.close()
                    pdf.close()

        paths = [simple_pdf, math_pdf] * ROUNDS
        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            sizes = list(pool.map(render, paths))
        assert all(size > 0 for size in sizes)

    def test_mixed_workload_is_safe(self, mixed_pdf, simple_pdf):
        """Triage and text extraction interleaved — the real batch pattern."""
        def work(index: int):
            path = mixed_pdf if index % 2 else simple_pdf
            decisions = triage_pdf(path)
            return extract_text_layer(path, [d.profile.index for d in decisions])

        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            outputs = list(pool.map(work, range(THREADS * ROUNDS)))
        assert all(out for out in outputs)


class TestBatchConcurrency:
    def test_batch_cli_converts_concurrently(self, tmp_path, simple_pdf, mixed_pdf):
        """Document-level concurrency: the knob that matters for one-page files.

        ``--workers`` only parallelises pages inside a document, so a corpus of
        single-page scans would run strictly serially without ``--doc-workers``.
        """
        from docfusion.cli import main as cli_main

        in_dir = tmp_path / "in"
        in_dir.mkdir()
        for i in range(12):
            source = simple_pdf if i % 2 else mixed_pdf
            (in_dir / f"doc{i:02d}.pdf").write_bytes(source.read_bytes())
        out_dir = tmp_path / "out"

        rc = cli_main(["batch", str(in_dir), str(out_dir),
                       "--tier1-only", "--no-docling", "--doc-workers", "8"])
        assert rc == 0
        assert len(list(out_dir.glob("*.md"))) == 12
        assert all(p.read_text(encoding="utf-8").strip() for p in out_dir.glob("*.md"))
