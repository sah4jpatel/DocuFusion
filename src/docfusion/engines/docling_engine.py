"""Tier-1 deterministic extraction.

Prefers Docling (MIT; DocLayNet layout + TableFormer) when installed. Falls
back to the raw pdfium text layer so the pipeline still works in minimal
environments — appropriate anyway, since heuristic triage only routes clean
born-digital pages here.

.. note::
   **Docling's OCR is disabled by default here.** Left on, Docling initialises
   RapidOCR, which downloads PP-OCR weights from ``modelscope.cn`` on first use.
   That is a third model family entering the runtime BOM from a host nobody
   audited, arriving silently at conversion time rather than at install time —
   the exact failure mode :mod:`docfusion.licenses` exists to prevent. It is
   also redundant: triage sends every page with a weak text layer to Tier 2, so
   a page that reaches Docling has a text layer worth trusting. Set
   ``PipelineConfig.docling_ocr=True`` to opt back in, and note the audit will
   then require the ``rapidocr`` component to be cleared explicitly.

Also exposes :func:`build_docling_vlm_options` for the alternative topology in
which Docling itself is the orchestrator and calls olmOCR through its native
``ApiVlmOptions`` (no Marker in the loop).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pypdfium2 as pdfium

from docfusion.config import VLMEndpoint
from docfusion.pdfium_lock import pdfium_guard

logger = logging.getLogger(__name__)


def docling_available() -> bool:
    try:
        import docling  # noqa: F401
        return True
    except ImportError:
        return False


def extract_text_layer(path: str | Path, page_indices: list[int]) -> dict[int, str]:
    """Minimal fallback: plain text layer per page (already-clean pages only)."""
    with pdfium_guard():
        pdf = pdfium.PdfDocument(str(path))
        try:
            out: dict[int, str] = {}
            for i in page_indices:
                page = pdf[i]
                tp = page.get_textpage()
                try:
                    out[i] = (tp.get_text_bounded() or "").strip()
                finally:
                    tp.close()
                    page.close()
            return out
        finally:
            pdf.close()


def _build_converter(enable_ocr: bool, device: str = "cpu"):
    """A DocumentConverter with OCR off and pinned to a device.

    ``device`` defaults to CPU because Tier 1 usually shares a box with the
    vLLM server, and vLLM pre-allocates ~90% of VRAM at startup. Docling then
    competes for the few hundred MB left, and the symptom is not a Docling
    error — it is Tier-2 requests timing out. Docling's layout and table models
    are small enough to run on CPU by design; that is the point of the tier.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = enable_ocr
    options.do_table_structure = True

    try:
        from docling.datamodel.accelerator_options import (
            AcceleratorDevice,
            AcceleratorOptions,
        )

        options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice(device))
    except Exception as exc:  # noqa: BLE001 — older Docling without accelerator options
        logger.debug("could not pin Docling device to %s: %s", device, exc)

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def extract_with_docling(
    path: str | Path, page_indices: list[int], enable_ocr: bool = False, device: str = "cpu"
) -> dict[int, str]:
    """Structured Markdown per page via Docling (import deferred)."""
    converter = _build_converter(enable_ocr, device)
    result = converter.convert(str(path))
    doc = result.document
    out: dict[int, str] = {}
    for i in page_indices:
        # Docling pages are 1-indexed.
        try:
            out[i] = doc.export_to_markdown(page_no=i + 1)
        except Exception as exc:  # noqa: BLE001 — a page-level export failure is recoverable
            logger.warning("docling export failed for page %d of %s: %s", i, path, exc)
            out[i] = ""
    return out


def tier1_extract(
    path: str | Path,
    page_indices: list[int],
    prefer_docling: bool = True,
    enable_ocr: bool = False,
    device: str = "cpu",
) -> dict[int, str]:
    """Extract Tier-1 pages, degrading to the text layer if Docling fails.

    Docling loads layout and table models and can fail on a malformed document.
    Falling back keeps a single bad file from taking out a batch run.
    """
    if not page_indices:
        return {}
    if prefer_docling and docling_available():
        try:
            return extract_with_docling(path, page_indices, enable_ocr=enable_ocr, device=device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("docling failed on %s (%s); using raw text layer", path, exc)
    return extract_text_layer(path, page_indices)


def build_docling_vlm_options(endpoint: VLMEndpoint | None = None):
    """Docling-as-orchestrator: VlmPipelineOptions pointed at local olmOCR.

    The prompt and response format are taken from the olmOCR-2 contract rather
    than invented, so this topology and the Marker one query the model
    identically. Docling receives Markdown; olmOCR-2 prefixes it with YAML front
    matter, so callers must strip it with
    :func:`docfusion.engines.olmocr_protocol.split_front_matter`.
    """
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat

    from docfusion.engines.olmocr_protocol import V4_YAML_PROMPT

    ep = endpoint or VLMEndpoint()
    vlm_options = ApiVlmOptions(
        url=f"{ep.base_url}/chat/completions",
        params={"model": ep.model, "max_tokens": ep.max_output_tokens, "temperature": 0.1},
        prompt=V4_YAML_PROMPT,
        timeout=ep.timeout_s,
        response_format=ResponseFormat.MARKDOWN,
    )
    opts = VlmPipelineOptions(enable_remote_services=True)
    opts.vlm_options = vlm_options
    return opts
