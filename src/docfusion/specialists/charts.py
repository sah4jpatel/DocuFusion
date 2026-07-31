"""Chart derendering: read the numbers *out* of a plot.

This is the one gap no amount of prompting fixes on a generalist OCR model.
olmOCR-2's trained prompt tells it to emit a figure placeholder —
``![Alt text ...](page_x_y_w_h.png)`` — so a bar chart becomes a caption, and a
benchmark asking "what is the value of series IF at category '193 UN Member
States'?" has nothing to match. Every Markdown-linearising model on the
ParseBench leaderboard scores near zero on charts for the same reason:
Dots.mocr 0.95, DeepSeek-OCR-2 1.1, PaddleOCR-VL 0.9.

DePlot solves exactly this and nothing else. It is a Pix2Struct model trained
to *derender* a plot into the data table that produced it, it is Apache-2.0
from Google, and at ~1.3 GB it is a rounding error next to the 7B generalist.

Its output is a flat linearised table using ``<0x0A>`` as the row separator and
``|`` as the cell separator, which this module converts to a Markdown table so
the value, its series and its axis label all end up adjacent in the text — the
form the chart tests actually look for.
"""

from __future__ import annotations

import logging
import os
import re

from docfusion.specialists.base import (
    Region,
    RegionKind,
    SpecialistResult,
    register_specialist,
)

logger = logging.getLogger(__name__)

DEPLOT_MODEL = os.getenv("DOCFUSION_CHART_MODEL", "google/deplot")
DEPLOT_PROMPT = "Generate underlying data table of the figure below:"

# DePlot emits literal "<0x0A>" between rows rather than a newline.
_ROW_SEPARATOR = re.compile(r"<0x0A>|\n")
_TITLE_RE = re.compile(r"^\s*title\s*\|\s*(.+)$", re.IGNORECASE)


def linearised_to_markdown(raw: str) -> str:
    """Turn DePlot's flat output into a Markdown table.

    A Markdown table is chosen over prose because it keeps each value next to
    both of its labels. Flattening to a sentence would separate "0.8079" from
    the series and category it belongs to, which is the whole content of a
    chart data point.
    """
    if not raw or not raw.strip():
        return ""

    rows: list[list[str]] = []
    title = ""
    for line in _ROW_SEPARATOR.split(raw):
        line = line.strip()
        if not line:
            continue
        match = _TITLE_RE.match(line)
        if match:
            title = match.group(1).strip()
            continue
        cells = [c.strip() for c in line.split("|")]
        if any(cells):
            rows.append(cells)

    if not rows:
        return f"**{title}**" if title else ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    header, body = rows[0], rows[1:]
    lines: list[str] = []
    if title:
        lines.append(f"**{title}**")
        lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * width) + "|")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


class DePlotSpecialist:
    """Chart → data table (Apache-2.0, Google Research)."""

    name = "deplot"
    kinds = (RegionKind.CHART, RegionKind.FIGURE)
    licence = "Apache-2.0"
    origin = "Google Research (US)"

    def __init__(self, model_name: str = DEPLOT_MODEL, device: str | None = None,
                 max_new_tokens: int = 512):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._device = device
        self._model = None
        self._processor = None

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = Pix2StructProcessor.from_pretrained(self.model_name)
        model = Pix2StructForConditionalGeneration.from_pretrained(self.model_name)
        self._model = model.to(device).eval()
        self._device = device
        logger.info("loaded %s on %s", self.model_name, device)

    def run(self, region: Region) -> SpecialistResult:
        if region.image is None:
            return SpecialistResult(specialist=self.name, note="no image supplied")
        try:
            self._load()
            import torch

            inputs = self._processor(
                images=region.image, text=DEPLOT_PROMPT, return_tensors="pt"
            ).to(self._device)
            with torch.inference_mode():
                predictions = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            raw = self._processor.decode(predictions[0], skip_special_tokens=True)
        except Exception as exc:  # noqa: BLE001 — a failed chart must not fail the page
            logger.warning("deplot failed on page %d region %d: %s",
                           region.page_index, region.reading_order, exc)
            return SpecialistResult(specialist=self.name, degraded=True, note=str(exc))

        markdown = linearised_to_markdown(raw)
        return SpecialistResult(
            markdown=markdown,
            specialist=self.name,
            note="" if markdown else "empty derendering",
        )


@register_specialist(
    "deplot",
    kinds=(RegionKind.CHART, RegionKind.FIGURE),
    licence="Apache-2.0",
    origin="Google Research (US)",
)
def _make_deplot() -> DePlotSpecialist:
    return DePlotSpecialist()
