"""Model-free page triage.

Profiles each PDF page from its raw structure (text layer, page objects) and
decides FAST (deterministic Tier 1) vs VLM (Tier 2 escalation). This runs at
thousands of pages/sec on CPU and requires no model downloads, so it can sit in
front of both Docling and the VLM regardless of environment.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pypdfium2 as pdfium

from docfusion.config import TriageThresholds
from docfusion.pdfium_lock import pdfium_guard

logger = logging.getLogger(__name__)

# Characters strongly associated with mathematical typesetting.
_MATH_CHARS = set("∑∏∫√∞≈≠≤≥±×÷∂∇∈∉⊂⊃∪∩∀∃αβγδεζηθλμνξπρστφχψωΓΔΘΛΞΠΣΦΨΩ^")
_MATH_RE = re.compile(r"[=+\-*/^_{}\\]")


class Route(str, Enum):
    FAST = "fast"   # Tier 1: deterministic (Docling / text layer)
    VLM = "vlm"     # Tier 2: Marker + olmOCR escalation


@dataclass
class PageProfile:
    index: int
    char_count: int
    word_count: int
    avg_word_len: float
    math_density: float
    image_area_ratio: float
    path_object_count: int
    image_object_count: int


@dataclass
class PageDecision:
    profile: PageProfile
    route: Route
    reasons: list[str]


def profile_page(page: pdfium.PdfPage, index: int) -> PageProfile:
    textpage = page.get_textpage()
    try:
        text = textpage.get_text_bounded() or ""
    finally:
        textpage.close()

    words = [w for w in re.split(r"\s+", text) if w]
    char_count = len(text.strip())
    avg_word_len = (sum(len(w) for w in words) / len(words)) if words else 0.0

    math_hits = sum(1 for ch in text if ch in _MATH_CHARS) + len(_MATH_RE.findall(text))
    math_density = math_hits / char_count if char_count else 0.0

    page_area = max(page.get_width() * page.get_height(), 1.0)
    image_area = 0.0
    n_paths = 0
    n_images = 0
    for obj in page.get_objects(max_depth=2):
        if obj.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE:
            n_images += 1
            try:
                bounds = obj.get_bounds()  # (left, bottom, right, top)
                left, bottom, right, top = bounds
                image_area += max(right - left, 0) * max(top - bottom, 0)
            except (pdfium.PdfiumError, AttributeError, TypeError, ValueError):
                pass
        elif obj.type == pdfium.raw.FPDF_PAGEOBJ_PATH:
            n_paths += 1

    return PageProfile(
        index=index,
        char_count=char_count,
        word_count=len(words),
        avg_word_len=avg_word_len,
        math_density=math_density,
        image_area_ratio=min(image_area / page_area, 1.0),
        path_object_count=n_paths,
        image_object_count=n_images,
    )


def decide(profile: PageProfile, t: TriageThresholds) -> PageDecision:
    reasons: list[str] = []
    if profile.char_count < t.min_text_chars:
        reasons.append(f"sparse text layer ({profile.char_count} chars < {t.min_text_chars}) — likely scan")
    if profile.image_area_ratio > t.max_image_area_ratio:
        reasons.append(f"image coverage {profile.image_area_ratio:.0%} > {t.max_image_area_ratio:.0%}")
    if profile.math_density > t.max_math_density:
        reasons.append(f"math density {profile.math_density:.3f} > {t.max_math_density}")
    if profile.path_object_count > t.max_path_objects:
        reasons.append(f"{profile.path_object_count} vector paths > {t.max_path_objects} — complex tables/figures")
    if 0 < profile.word_count and profile.avg_word_len < t.min_avg_word_len:
        reasons.append(f"avg word length {profile.avg_word_len:.1f} < {t.min_avg_word_len} — garbled text layer")
    if profile.char_count > t.max_chars_per_page:
        reasons.append(f"{profile.char_count} chars > {t.max_chars_per_page} — very dense page (tiny text)")

    return PageDecision(
        profile=profile,
        route=Route.VLM if reasons else Route.FAST,
        reasons=reasons,
    )


def _unreadable_page(index: int, why: str) -> PageDecision:
    """A page we cannot profile is escalated, never silently dropped.

    Failing to parse the structure is itself evidence the text layer is
    untrustworthy, which is exactly the case Tier 2 exists for.
    """
    profile = PageProfile(
        index=index, char_count=0, word_count=0, avg_word_len=0.0,
        math_density=0.0, image_area_ratio=0.0, path_object_count=0,
        image_object_count=0,
    )
    return PageDecision(profile=profile, route=Route.VLM, reasons=[f"page unreadable ({why})"])


def triage_pdf(path: str | Path, thresholds: TriageThresholds | None = None,
               password: str | None = None) -> list[PageDecision]:
    """Profile and route every page. Never raises on a per-page defect."""
    t = thresholds or TriageThresholds()
    with pdfium_guard():
        return _triage_locked(path, t, password)


def _triage_locked(path, t, password):
    pdf = pdfium.PdfDocument(str(path), password=password)
    try:
        decisions: list[PageDecision] = []
        for i in range(len(pdf)):
            page = None
            try:
                page = pdf[i]
                decisions.append(decide(profile_page(page, i), t))
            except Exception as exc:  # noqa: BLE001 — one bad page must not kill the document
                logger.warning("triage failed on page %d of %s: %s", i, path, exc)
                decisions.append(_unreadable_page(i, f"{type(exc).__name__}: {exc}"))
            finally:
                if page is not None:
                    try:
                        page.close()
                    except Exception:  # noqa: BLE001
                        pass
        return decisions
    finally:
        pdf.close()
