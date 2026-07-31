"""Region router: segment a page, send each region to whatever is best at it.

The generalist reads the page. This module finds the parts of the page the
generalist is known to handle badly — charts it labels instead of reading,
display formulas on scans — crops them, sends them to a specialist, and splices
the result back in.

Detection is deliberately cheap and model-free by default. A chart in a
born-digital PDF is a dense cluster of vector paths with almost no text on top
of it; an embedded figure is an image object. Both are visible in the PDF's own
object list, which costs about a millisecond to walk. Docling's layout model is
used instead when it is installed and enabled, because it classifies regions
properly rather than inferring from geometry — but requiring it would make the
whole feature conditional on a GPU-class dependency, and the heuristic is good
enough to route with.

Nothing here fails closed. A region whose specialist is missing keeps the
generalist's text for that area, so enabling fusion can add information but
cannot remove any.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import pypdfium2 as pdfium

from docfusion.pdfium_lock import pdfium_guard
from docfusion.specialists.base import (
    FusionReport,
    Region,
    RegionKind,
    SpecialistResult,
    available_specialists,
)

logger = logging.getLogger(__name__)

# A chart is drawn, not written: many path objects, little text over them.
# Path count alone is not enough — a ruled table is also many paths, and
# sending one to a chart model wastes a GPU pass on something Tier 1 already
# handles. Text density separates them: a chart is mostly ink, a table is
# mostly characters.
MIN_CHART_PATHS = 12
# Characters per unit of normalised page area. Measured over 45 chart regions in
# ParseBench: median 2203, p90 4545. Density alone is a weak discriminator at the
# boundary — a dense ruled table lands inside the chart range — so it only rules
# out the extremes, and the lattice test below does the real work.
MAX_CHART_TEXT_DENSITY = 5000.0
# A ruled table is a lattice: long lines in BOTH axes forming cells. A chart has
# gridlines in one axis and bars in the other, so requiring both is what
# separates them. This is structural rather than statistical, which is why it
# holds where the density threshold does not.
LATTICE_MIN_RULES_PER_AXIS = 3
LATTICE_RULE_SPAN = 0.6      # fraction of the region a line must cross to count
# ...and it must be THIN. A bar in a column chart is tall enough to span the
# plot area, so length alone counts bars as vertical rules and rejects the very
# charts this is meant to find. A rule is a line; a bar is a shape.
LATTICE_MAX_RULE_THICKNESS = 3.0     # PDF points
# Ignore decorations — a region must occupy a meaningful slice of the page.
MIN_REGION_AREA = 0.015
# Crops are rendered generously; chart models are sensitive to clipped axes.
REGION_PADDING = 0.012
CHART_CROP_LONGEST_DIM = 900


@dataclass
class DetectedRegion:
    kind: RegionKind
    x: float
    y: float
    w: float
    h: float
    path_count: int = 0
    text_chars: int = 0

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)

    @property
    def text_density(self) -> float:
        """Characters per unit of normalised page area."""
        return self.text_chars / self.area if self.area > 0 else 0.0


@dataclass
class FusionResult:
    markdown: str
    report: FusionReport = field(default_factory=FusionReport)
    regions: list[DetectedRegion] = field(default_factory=list)


def _char_boxes(page: pdfium.PdfPage) -> list[tuple[float, float]]:
    """Centre point of every character on the page, in PDF user space.

    Centres rather than boxes: deciding whether a glyph is "inside" a region
    only needs one point, and this runs once per page over thousands of chars.
    Caller must already hold the PDFium lock.
    """
    import ctypes

    import pypdfium2.raw as raw

    textpage = page.get_textpage()
    points: list[tuple[float, float]] = []
    try:
        count = raw.FPDFText_CountChars(textpage.raw)
        left = ctypes.c_double(0)
        right = ctypes.c_double(0)
        bottom = ctypes.c_double(0)
        top = ctypes.c_double(0)
        for index in range(count):
            if raw.FPDFText_GetCharBox(
                textpage.raw, index,
                ctypes.byref(left), ctypes.byref(right),
                ctypes.byref(bottom), ctypes.byref(top),
            ):
                points.append(
                    ((left.value + right.value) / 2.0, (bottom.value + top.value) / 2.0)
                )
    except Exception:  # noqa: BLE001 — a page without a text layer has no chars
        return points
    finally:
        textpage.close()
    return points


def _looks_like_lattice(
    paths: list[tuple[float, float, float, float]],
    box: tuple[float, float, float, float],
) -> bool:
    """True when the paths inside ``box`` form a ruled grid rather than a plot."""
    x0, y0, x1, y1 = box
    width = max(x1 - x0, 1e-6)
    height = max(y1 - y0, 1e-6)
    horizontal = vertical = 0
    for px0, py0, px1, py1 in paths:
        if px1 < x0 or px0 > x1 or py1 < y0 or py0 > y1:
            continue
        thickness_y = py1 - py0
        thickness_x = px1 - px0
        is_h_rule = (thickness_x >= width * LATTICE_RULE_SPAN
                     and thickness_y <= LATTICE_MAX_RULE_THICKNESS)
        is_v_rule = (thickness_y >= height * LATTICE_RULE_SPAN
                     and thickness_x <= LATTICE_MAX_RULE_THICKNESS)
        if is_h_rule:
            horizontal += 1
        elif is_v_rule:
            vertical += 1
    return (horizontal >= LATTICE_MIN_RULES_PER_AXIS
            and vertical >= LATTICE_MIN_RULES_PER_AXIS)


def _cluster(boxes: list[tuple[float, float, float, float]],
             gap: float) -> list[tuple[float, float, float, float]]:
    """Merge boxes that touch or nearly touch, repeatedly until stable.

    Charts arrive as hundreds of separate path objects — one per bar, gridline
    and tick. Individually none of them is a region; together they are one.
    """
    merged = list(boxes)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        out: list[tuple[float, float, float, float]] = []
        while merged:
            x0, y0, x1, y1 = merged.pop()
            keep: list[tuple[float, float, float, float]] = []
            for other in merged:
                ox0, oy0, ox1, oy1 = other
                if (x0 - gap <= ox1 and ox0 <= x1 + gap
                        and y0 - gap <= oy1 and oy0 <= y1 + gap):
                    x0, y0 = min(x0, ox0), min(y0, oy0)
                    x1, y1 = max(x1, ox1), max(y1, oy1)
                    changed = True
                else:
                    keep.append(other)
            merged = keep
            out.append((x0, y0, x1, y1))
        merged = out
    return merged


def detect_regions(page: pdfium.PdfPage) -> list[DetectedRegion]:
    """Find figure/chart regions from the PDF's own object list."""
    with pdfium_guard():
        width = float(page.get_width())
        height = float(page.get_height())
        if width <= 0 or height <= 0:
            return []
        try:
            objects = list(page.get_objects(max_depth=2))
        except Exception:  # noqa: BLE001
            return []

        images: list[tuple[float, float, float, float]] = []
        paths: list[tuple[float, float, float, float]] = []
        for obj in objects:
            try:
                bounds = obj.get_bounds()
            except Exception:  # noqa: BLE001
                continue
            if obj.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE:
                images.append(bounds)
            elif obj.type == pdfium.raw.FPDF_PAGEOBJ_PATH:
                paths.append(bounds)

        char_boxes = _char_boxes(page)

    regions: list[DetectedRegion] = []

    def chars_within(box: tuple[float, float, float, float]) -> int:
        x0, y0, x1, y1 = box
        return sum(
            1 for cx, cy in char_boxes
            if x0 <= cx <= x1 and y0 <= cy <= y1
        )

    def add(kind: RegionKind, box: tuple[float, float, float, float], n_paths: int = 0) -> None:
        x0, y0, x1, y1 = box
        region = DetectedRegion(
            kind=kind,
            x=max(x0 / width, 0.0),
            # PDF space is bottom-left; regions are reported top-left like the
            # rest of the pipeline.
            y=max((height - y1) / height, 0.0),
            w=min((x1 - x0) / width, 1.0),
            h=min((y1 - y0) / height, 1.0),
            path_count=n_paths,
            text_chars=chars_within(box),
        )
        if region.area < MIN_REGION_AREA:
            return
        if kind is RegionKind.CHART and region.text_density > MAX_CHART_TEXT_DENSITY:
            # Dense with characters: this is a ruled table, not a plot.
            return
        regions.append(region)

    for box in images:
        add(RegionKind.FIGURE, box)

    # Vector clusters: a chart is many small paths in one area.
    if len(paths) >= MIN_CHART_PATHS:
        gap = min(width, height) * 0.02
        for cluster in _cluster(paths, gap):
            members = sum(
                1 for p in paths
                if p[0] >= cluster[0] - gap and p[2] <= cluster[2] + gap
                and p[1] >= cluster[1] - gap and p[3] <= cluster[3] + gap
            )
            if members < MIN_CHART_PATHS:
                continue
            if _looks_like_lattice(paths, cluster):
                continue          # ruled table: Tier 1 already handles this well
            add(RegionKind.CHART, cluster, members)

    return regions


def crop_region(page: pdfium.PdfPage, region: DetectedRegion,
                longest_dim: int = CHART_CROP_LONGEST_DIM):
    """Render just this region, with a little padding around it."""
    from PIL import Image  # noqa: F401  (import cost only when fusion is used)

    with pdfium_guard():
        width = float(page.get_width())
        height = float(page.get_height())
        scale = longest_dim / max(width, height, 1.0)
        pil = page.render(scale=scale).to_pil()

    px, py = pil.size
    pad = REGION_PADDING
    left = int(max(region.x - pad, 0.0) * px)
    top = int(max(region.y - pad, 0.0) * py)
    right = int(min(region.x + region.w + pad, 1.0) * px)
    bottom = int(min(region.y + region.h + pad, 1.0) * py)
    if right - left < 8 or bottom - top < 8:
        return None
    return pil.crop((left, top, right, bottom))


def fuse_page(
    page: pdfium.PdfPage,
    page_index: int,
    page_markdown: str,
    kinds: tuple[RegionKind, ...] = (RegionKind.CHART, RegionKind.FIGURE),
) -> FusionResult:
    """Run specialists over a page's regions and append what they recover.

    Specialist output is *appended*, not substituted for the generalist's text.
    The generalist may already have transcribed a chart's axis labels usefully,
    and a chart model's table is additional information rather than a
    correction — deleting text to make room for it could only lose content.
    """
    report = FusionReport()
    regions = [r for r in detect_regions(page) if r.kind in kinds]
    report.regions = len(regions)
    for region in regions:
        report.by_kind[region.kind.value] = report.by_kind.get(region.kind.value, 0) + 1

    if not regions:
        return FusionResult(markdown=page_markdown, report=report)

    additions: list[str] = []
    for order, region in enumerate(sorted(regions, key=lambda r: (r.y, r.x))):
        specialists = available_specialists(region.kind)
        if not specialists:
            report.fell_back += 1
            continue

        image = crop_region(page, region)
        if image is None:
            report.fell_back += 1
            continue

        payload = Region(
            kind=region.kind, page_index=page_index,
            x=region.x, y=region.y, w=region.w, h=region.h,
            image=image, reading_order=order,
        )
        result: SpecialistResult | None = None
        for name, specialist in specialists.items():
            result = specialist.run(payload)
            if result.usable:
                report.by_specialist[name] = report.by_specialist.get(name, 0) + 1
                break
        if result is None or not result.usable:
            report.fell_back += 1
            continue
        additions.append(result.markdown)

    if not additions:
        return FusionResult(markdown=page_markdown, report=report, regions=regions)

    body = page_markdown.rstrip()
    joined = "\n\n".join(additions)
    return FusionResult(
        markdown=f"{body}\n\n{joined}" if body else joined,
        report=report,
        regions=regions,
    )


def render_region_png(page: pdfium.PdfPage, region: DetectedRegion) -> bytes:
    """A region crop as PNG bytes — for debugging what the router selected."""
    image = crop_region(page, region)
    if image is None:
        return b""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
