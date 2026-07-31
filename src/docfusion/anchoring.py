"""Document anchoring: the born-digital text layer with coordinates, injected
into a VLM prompt as a deterministic prior.

.. warning::
   **This is not used for olmOCR-2 and must not be enabled for it.** Anchoring
   was the olmOCR *v1* technique; the 1025 release is trained on
   ``build_no_anchoring_v4_yaml_prompt()`` and upstream's own
   ``--target_anchor_text_len`` flag is documented "not used for new models".
   Feeding anchors to olmOCR-2 puts it off-distribution and wastes context that
   the page image needs.

   The module is retained because it remains correct for the anchoring-era
   models (olmOCR 0725/0825) and for other VLMs behind the same endpoint, and
   because the extracted text layer is what the Tier-2 fallback path emits when
   the model is unreachable. Gate it with ``VLMEndpoint.use_anchoring``.
"""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium

from docfusion.pdfium_lock import pdfium_guard

ANCHOR_HEADER = (
    "Below is the raw text layer extracted from this page with [x,y] anchor "
    "coordinates (origin bottom-left, PDF points). Treat it as ground truth for "
    "character content; use the image only for reading order, layout, tables, "
    "figures and equations. Do not invent text that appears in neither source."
)

PAGE_PROMPT = (
    "Convert this document page to clean GitHub-flavored Markdown in natural "
    "reading order. Render equations as LaTeX ($...$ / $$...$$), tables as "
    "Markdown tables preserving row/column spans as best as possible, and "
    "describe images as figure captions. Exclude running headers, footers and "
    "page numbers. Output only the Markdown."
)


def extract_anchors(page: pdfium.PdfPage, max_chars: int = 6000) -> str:
    """Return anchor lines like `[72,701] Quarterly revenue grew...`."""
    with pdfium_guard():
        return _extract_anchors_locked(page, max_chars)


def _extract_anchors_locked(page: pdfium.PdfPage, max_chars: int) -> str:
    textpage = page.get_textpage()
    try:
        n_rects = textpage.count_rects()
        lines: list[str] = []
        used = 0
        for i in range(n_rects):
            left, bottom, right, top = textpage.get_rect(i)
            chunk = textpage.get_text_bounded(left=left, bottom=bottom, right=right, top=top)
            chunk = (chunk or "").strip()
            if not chunk:
                continue
            line = f"[{left:.0f},{bottom:.0f}] {chunk}"
            used += len(line) + 1
            if used > max_chars:
                lines.append("[...anchor text truncated...]")
                break
            lines.append(line)
        return "\n".join(lines)
    finally:
        textpage.close()


def build_anchored_prompt(page: pdfium.PdfPage, max_chars: int = 6000) -> str:
    anchors = extract_anchors(page, max_chars=max_chars)
    if not anchors:
        return PAGE_PROMPT  # scanned page: nothing to anchor, pure visual OCR
    return f"{PAGE_PROMPT}\n\n{ANCHOR_HEADER}\n\n{anchors}"


def anchored_prompt_for(path: str | Path, page_index: int, max_chars: int = 6000) -> str:
    with pdfium_guard():
        pdf = pdfium.PdfDocument(str(path))
        try:
            page = pdf[page_index]
            try:
                return build_anchored_prompt(page, max_chars=max_chars)
            finally:
                page.close()
        finally:
            pdf.close()
