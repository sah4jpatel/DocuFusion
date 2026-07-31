"""DocFusion dual-tier pipeline.

triage (model-free) ──► Tier 1: Docling / text layer   (~80% of pages, ~1.6 GB VRAM)
                   └──► Tier 2: olmOCR-2 via vLLM      (math / scans / dense tables)

Every run is preceded by a license audit of the components about to be loaded,
so a restricted model can never silently enter the BOM.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pypdfium2 as pdfium

from docfusion.config import PipelineConfig
from docfusion.engines.docling_engine import tier1_extract
from docfusion.engines.olmocr_client import OlmOCRClient, PageResult
from docfusion.formatting import FormattingReport, format_page_markdown
from docfusion.grounding import PageLayout, document_layout
from docfusion.io import as_pdf, sanitize_text
from docfusion.licenses import assert_compliant
from docfusion.pdfium_lock import pdfium_guard
from docfusion.triage.heuristics import PageDecision, Route, triage_pdf

logger = logging.getLogger(__name__)


@dataclass
class DocumentResult:
    path: str
    markdown: str
    decisions: list[PageDecision]
    tier2_pages: list[int] = field(default_factory=list)
    degraded_pages: list[int] = field(default_factory=list)
    fallback_pages: list[int] = field(default_factory=list)
    page_results: dict[int, PageResult] = field(default_factory=dict)
    page_markdown: dict[int, str] = field(default_factory=dict)
    formatting: dict[int, FormattingReport] = field(default_factory=dict)
    layout: list[PageLayout] = field(default_factory=list)

    @property
    def formatting_marks_applied(self) -> int:
        return sum(r.spans_applied for r in self.formatting.values())

    @property
    def page_count(self) -> int:
        return len(self.decisions)

    @property
    def tier2_fraction(self) -> float:
        return len(self.tier2_pages) / len(self.decisions) if self.decisions else 0.0

    @property
    def input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.page_results.values())

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.page_results.values())

    def summary(self) -> dict[str, object]:
        return {
            "path": self.path,
            "pages": self.page_count,
            "tier2_pages": len(self.tier2_pages),
            "tier2_fraction": round(self.tier2_fraction, 4),
            "degraded_pages": len(self.degraded_pages),
            "fallback_pages": len(self.fallback_pages),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "chars": len(self.markdown),
            "formatting_marks": self.formatting_marks_applied,
            "layout_blocks": sum(len(p.blocks) for p in self.layout),
        }


class DocFusionPipeline:
    def __init__(self, config: PipelineConfig | None = None,
                 vlm_client: OlmOCRClient | None = None):
        self.config = config or PipelineConfig()
        if self.config.enforce_license_audit:
            assert_compliant()
        self._vlm = vlm_client  # lazy: only built if a page actually escalates

    @property
    def vlm(self) -> OlmOCRClient:
        if self._vlm is None:
            self._vlm = OlmOCRClient(self.config.vlm)
        return self._vlm

    def _run_tier2(self, path: Path, page_indices: list[int]) -> dict[int, PageResult]:
        """OCR the escalated pages, several in flight at once.

        vLLM batches continuously, so a serial client leaves most of the GPU
        idle; each page is independent, so this is embarrassingly parallel.

        PDFium itself is serialised by :mod:`docfusion.pdfium_lock` — a
        per-thread ``PdfDocument`` is *not* sufficient isolation, because the
        library keeps global state and concurrent calls abort the process with
        a native allocator error. Only the inference wait actually overlaps,
        which is the part worth overlapping.
        """
        if not page_indices:
            return {}

        workers = min(self.config.max_tier2_workers, len(page_indices))

        def ocr_one(index: int) -> tuple[int, PageResult]:
            with pdfium_guard():
                pdf = pdfium.PdfDocument(str(path))
                page = pdf[index]
            try:
                # ocr_page takes the lock itself around each render.
                return index, self.vlm.ocr_page(page, index)
            finally:
                with pdfium_guard():
                    page.close()
                    pdf.close()

        if workers == 1:
            return dict(ocr_one(i) for i in page_indices)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return dict(pool.map(ocr_one, page_indices))

    # Markup a page's Markdown may already carry, in which case the text layer
    # has nothing to add and re-marking would only risk double emphasis.
    _EXISTING_MARKUP = ("**", "<u>", "~~", "\n#", "<b>", "<strong>", "<em>")

    def _recover_formatting(
        self, path: Path, page_markdown: dict[int, str]
    ) -> dict[int, FormattingReport]:
        """Re-apply typography from the text layer, page by page.

        Mutates ``page_markdown`` in place. Pages whose Markdown already carries
        markup are left alone — that is Docling's output, which encodes its own
        structure. Scanned pages have no font metadata and come back unchanged.
        """
        if not self.config.recover_formatting or not page_markdown:
            return {}

        reports: dict[int, FormattingReport] = {}
        candidates = [
            index for index, text in page_markdown.items()
            if text.strip() and not any(m in text for m in self._EXISTING_MARKUP)
            and not text.lstrip().startswith("#")
        ]
        if not candidates:
            return reports

        with pdfium_guard():
            pdf = pdfium.PdfDocument(str(path))
        try:
            for index in candidates:
                page = None
                try:
                    with pdfium_guard():
                        page = pdf[index]
                    marked, report = format_page_markdown(page, page_markdown[index])
                    page_markdown[index] = marked
                    reports[index] = report
                except Exception as exc:  # noqa: BLE001 — formatting is an enhancement
                    logger.warning("formatting recovery failed on page %d of %s: %s",
                                   index, path, exc)
                finally:
                    if page is not None:
                        with pdfium_guard():
                            page.close()
        finally:
            with pdfium_guard():
                pdf.close()
        return reports

    def convert(self, path: str | Path) -> DocumentResult:
        """Convert a document. Accepts PDFs and single-image scans alike."""
        original = Path(path)
        with as_pdf(original) as pdf_path:
            result = self._convert_pdf(pdf_path)
        # Report the path the caller gave us, not the temporary conversion.
        result.path = str(original)
        return result

    def _convert_pdf(self, path: Path) -> DocumentResult:
        decisions = triage_pdf(path, self.config.thresholds)
        escalated = [d.profile.index for d in decisions if d.route is Route.VLM]

        if self.config.force_tier2_all and self.config.tier2_enabled:
            # Accuracy ceiling: no triage, every page through the VLM.
            fast_pages: list[int] = []
            vlm_pages = [d.profile.index for d in decisions]
            escalated = vlm_pages
        elif self.config.tier2_enabled:
            fast_pages = [d.profile.index for d in decisions if d.route is Route.FAST]
            vlm_pages = escalated
        else:
            # Tier-1-only: everything goes deterministic, escalations are reported only.
            fast_pages = [d.profile.index for d in decisions]
            vlm_pages = []

        tier1 = tier1_extract(
            path, fast_pages,
            prefer_docling=self.config.use_docling_tier1,
            enable_ocr=self.config.docling_ocr,
            device=self.config.docling_device,
        )
        tier2 = self._run_tier2(path, vlm_pages)

        page_markdown: dict[int, str] = {}
        degraded: list[int] = []
        fallbacks: list[int] = []
        for decision in decisions:
            index = decision.profile.index
            if index in tier2:
                result = tier2[index]
                text = result.markdown
                if result.degraded:
                    degraded.append(index)
                if result.fallback:
                    fallbacks.append(index)
            else:
                text = tier1.get(index, "")
            page_markdown[index] = text

        formatting = self._recover_formatting(path, page_markdown)
        layout = document_layout(str(path)) if self.config.emit_layout else []
        parts = [page_markdown[d.profile.index] for d in decisions]

        return DocumentResult(
            path=str(path),
            markdown="\n\n".join(p for p in parts if p),
            decisions=decisions,
            tier2_pages=escalated,
            degraded_pages=degraded,
            fallback_pages=fallbacks,
            page_results=tier2,
            page_markdown={i: sanitize_text(t) for i, t in page_markdown.items()},
            formatting=formatting,
            layout=layout,
        )
