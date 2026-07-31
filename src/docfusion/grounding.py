"""Visual grounding: block bounding boxes and reading order from the text layer.

olmOCR-2 returns linearised Markdown and no coordinates, so anything that needs
to point at *where* a fact came from — citation highlighting, redaction review,
human verification of an extracted number — has nothing to point at. On
ParseBench's visual-grounding dimension that is a score of zero, and every other
Markdown-only parser on that leaderboard scores the same.

For a born-digital PDF the coordinates are not missing, only unused: PDFium
gives an exact box per glyph. Grouping those into lines and blocks reconstructs
the layout without a detector, without a GPU and without a model licence. What
it cannot do is read a scan — there is no text layer to group — so scanned pages
fall back to Docling's layout model, which is MIT-licensed and already in the
Tier-1 dependency set.

Coordinates are emitted normalised to ``[0, 1]`` with a **top-left origin**,
because that is the convention the consumers use. PDF user space is bottom-left,
so the y axis is flipped exactly once, here.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import pypdfium2 as pdfium

from docfusion.formatting import TextSpan, extract_spans
from docfusion.pdfium_lock import pdfium_guard

# Lines belong to the same block when they are close vertically relative to
# their own height. 1.8x leading comfortably spans double-spaced body text
# without swallowing the gap between a paragraph and the next heading.
BLOCK_LINE_GAP_RATIO = 1.8
# Two spans are on the same line when their vertical centres are within this
# fraction of line height.
LINE_TOLERANCE_RATIO = 0.55
# A horizontal gap this many multiples of the median character width apart
# suggests separate columns rather than one wide line.
COLUMN_GAP_RATIO = 6.0


@dataclass
class LayoutBlock:
    """One layout element with normalised, top-left-origin coordinates."""

    text: str
    x: float
    y: float
    w: float
    h: float
    label: str = "Text"
    reading_order: int = 0
    heading_level: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "type": self.label,
            "md": self.text,
            "value": self.text,
            "bbox": {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "label": self.label},
        }


@dataclass
class PageLayout:
    page_number: int                 # 1-indexed, as ParseBench expects
    width: float
    height: float
    blocks: list[LayoutBlock] = field(default_factory=list)
    source: str = "text_layer"       # or "docling"

    def as_dict(self) -> dict[str, object]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "text": "\n".join(b.text for b in self.blocks),
            "items": [b.as_dict() for b in self.blocks],
        }


def _split_on_gutters(line: list[TextSpan], gutter: float) -> list[list[TextSpan]]:
    """Break one visual row wherever a gutter-sized horizontal gap appears.

    Without this, a row of the left column and the row beside it in the right
    column share a vertical centre and merge into a single 'line' spanning the
    page — which then produces one full-width block instead of two, and reading
    order runs across the gutter instead of down the column.
    """
    if len(line) < 2:
        return [line]
    segments: list[list[TextSpan]] = []
    current = [line[0]]
    for span in line[1:]:
        if span.x0 - current[-1].x1 > gutter:
            segments.append(current)
            current = [span]
        else:
            current.append(span)
    segments.append(current)
    return segments


def _group_lines(spans: list[TextSpan], page_width: float = 0.0) -> list[list[TextSpan]]:
    """Cluster spans into visual lines, then split rows across column gutters."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (-(s.y0 + s.y1) / 2.0, s.x0))
    heights = [s.height for s in spans if s.height > 0] or [10.0]
    median_height = statistics.median(heights)
    tolerance = median_height * LINE_TOLERANCE_RATIO
    # A gutter is far wider than inter-word space; scale with type size, and
    # floor it against page width so tiny type cannot make every space a gutter.
    gutter = max(median_height * COLUMN_GAP_RATIO, page_width * 0.035 if page_width else 0.0)

    rows: list[list[TextSpan]] = []
    current: list[TextSpan] = [ordered[0]]
    current_y = (ordered[0].y0 + ordered[0].y1) / 2.0
    for span in ordered[1:]:
        centre = (span.y0 + span.y1) / 2.0
        if abs(centre - current_y) <= tolerance:
            current.append(span)
        else:
            rows.append(sorted(current, key=lambda s: s.x0))
            current = [span]
            current_y = centre
    rows.append(sorted(current, key=lambda s: s.x0))

    lines: list[list[TextSpan]] = []
    for row in rows:
        lines.extend(_split_on_gutters(row, gutter))
    return lines


def _line_box(line: list[TextSpan]) -> tuple[float, float, float, float]:
    return (
        min(s.x0 for s in line),
        min(s.y0 for s in line),
        max(s.x1 for s in line),
        max(s.y1 for s in line),
    )


def _split_columns(lines: list[list[TextSpan]], page_width: float) -> list[list[list[TextSpan]]]:
    """Split lines into columns when a consistent vertical gutter exists.

    Reading order is the whole point of grounding: a two-column page read
    straight across produces boxes in an order no consumer can follow. The test
    is deliberately conservative — a gutter must be wide, and must be clear on
    most lines — because falsely splitting a single-column page is worse than
    not splitting a two-column one.
    """
    if len(lines) < 6 or page_width <= 0:
        return [lines]

    mid = page_width / 2.0
    left_only = sum(1 for line in lines if _line_box(line)[2] <= mid + page_width * 0.02)
    right_only = sum(1 for line in lines if _line_box(line)[0] >= mid - page_width * 0.02)
    spanning = len(lines) - left_only - right_only

    # Both halves must be populated and few lines may cross the gutter.
    if left_only >= 3 and right_only >= 3 and spanning <= max(2, len(lines) * 0.15):
        left = [line for line in lines if _line_box(line)[2] <= mid + page_width * 0.02]
        right = [line for line in lines if _line_box(line)[0] >= mid - page_width * 0.02]
        crossing = [
            line for line in lines
            if line not in left and line not in right
        ]
        # Full-width lines (titles, rules) lead the page.
        return [crossing, left, right] if crossing else [left, right]
    return [lines]


def _lines_to_blocks(lines: list[list[TextSpan]]) -> list[tuple[list[list[TextSpan]], int]]:
    """Merge vertically adjacent lines of like style into blocks."""
    blocks: list[tuple[list[list[TextSpan]], int]] = []
    current: list[list[TextSpan]] = []
    current_level = 0

    for line in lines:
        level = max((s.heading_level for s in line), default=0)
        if not current:
            current, current_level = [line], level
            continue

        previous_box = _line_box(current[-1])
        box = _line_box(line)
        gap = previous_box[1] - box[3]            # previous bottom minus this top
        line_height = max(box[3] - box[1], 1.0)
        same_style = level == current_level
        overlapping = min(previous_box[2], box[2]) - max(previous_box[0], box[0]) > 0

        if same_style and overlapping and gap <= line_height * BLOCK_LINE_GAP_RATIO:
            current.append(line)
        else:
            blocks.append((current, current_level))
            current, current_level = [line], level
    if current:
        blocks.append((current, current_level))
    return blocks


def _label_for(level: int, text: str) -> str:
    if level == 1:
        return "Title"
    if level > 1:
        return "Section"
    stripped = text.lstrip()
    if stripped.startswith(("- ", "* ", "• ")) or (stripped[:2].rstrip(".").isdigit() and ". " in stripped[:4]):
        return "List"
    return "Text"


def page_layout_from_text_layer(
    page: pdfium.PdfPage, page_number: int, spans: list[TextSpan] | None = None
) -> PageLayout:
    """Reconstruct layout blocks for one page from its text layer."""
    with pdfium_guard():
        width = float(page.get_width())
        height = float(page.get_height())

    if spans is None:
        spans = extract_spans(page)

    layout = PageLayout(page_number=page_number, width=width, height=height)
    if not spans or width <= 0 or height <= 0:
        return layout

    lines = _group_lines(spans, width)
    order = 0
    for column in _split_columns(lines, width):
        for line_group, level in _lines_to_blocks(column):
            flat = [s for line in line_group for s in line]
            if not flat:
                continue
            text = " ".join(
                " ".join(s.text for s in line).strip() for line in line_group
            ).strip()
            if not text:
                continue
            x0 = min(s.x0 for s in flat)
            y0 = min(s.y0 for s in flat)
            x1 = max(s.x1 for s in flat)
            y1 = max(s.y1 for s in flat)
            layout.blocks.append(
                LayoutBlock(
                    text=text,
                    x=max(x0 / width, 0.0),
                    # PDF user space is bottom-left; consumers expect top-left.
                    y=max((height - y1) / height, 0.0),
                    w=min((x1 - x0) / width, 1.0),
                    h=min((y1 - y0) / height, 1.0),
                    label=_label_for(level, text),
                    reading_order=order,
                    heading_level=level,
                )
            )
            order += 1
    return layout


def document_layout(path: str, page_indices: list[int] | None = None) -> list[PageLayout]:
    """Layout for every requested page of a document."""
    with pdfium_guard():
        pdf = pdfium.PdfDocument(str(path))
    try:
        indices = page_indices if page_indices is not None else list(range(len(pdf)))
        layouts: list[PageLayout] = []
        for index in indices:
            page = None
            try:
                with pdfium_guard():
                    page = pdf[index]
                layouts.append(page_layout_from_text_layer(page, index + 1))
            finally:
                if page is not None:
                    with pdfium_guard():
                        page.close()
        return layouts
    finally:
        with pdfium_guard():
            pdf.close()
