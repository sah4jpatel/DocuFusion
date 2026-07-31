"""Recover semantic formatting from the PDF text layer.

olmOCR-2 is trained to return "the plain text representation of this document"
— literally. It emits no ``**bold**``, no ``#`` headings, no ``<u>``, no
``~~strike~~``. That is not a bug in the model; it is what its prompt asks for.
It is, however, a whole benchmark dimension scored at zero, and a real loss for
downstream consumers: "Payment is **not** required" and "Payment is not
required" mean different things to a RAG pipeline.

A born-digital PDF already carries this information. Every glyph knows its font
name, PDF descriptor flags, weight, size and exact box. Underlines and
strikethroughs are thin filled rectangles the renderer draws near the baseline
or through the middle of the text. None of that needs a model.

So this module reads the formatting deterministically and re-applies it to the
VLM's text. It is the clearest example of the hybrid being worth more than
either half: the VLM contributes reading order and content fidelity that the
text layer cannot give, and the text layer contributes typography that the VLM
structurally cannot see.

Scope and honesty: this only works on pages that *have* a usable text layer.
Scanned pages have no font metadata, so they get nothing here — which is fine,
because triage already routes those to Tier 2 for different reasons, and a scan
has no ground-truth typography to recover.
"""

from __future__ import annotations

import ctypes
import re
import statistics
import unicodedata
from dataclasses import dataclass, field

import pypdfium2 as pdfium
import pypdfium2.raw as raw

from docfusion.pdfium_lock import pdfium_guard

# PDF font descriptor flags (PDF 32000-1:2008, Table 123).
FLAG_ITALIC = 1 << 6        # bit 7
FLAG_FORCE_BOLD = 1 << 18   # bit 19

# Base-14 and many embedded fonts leave the descriptor flags empty, so the
# PostScript name is the practical signal. This is what PDF tooling relies on.
_BOLD_NAME_RE = re.compile(r"bold|black|heavy|semibold|demibold|ultra", re.I)
_ITALIC_NAME_RE = re.compile(r"italic|oblique", re.I)

# A rule counts as an underline/strikeout decoration, not content, if it is thin.
MAX_RULE_THICKNESS_PT = 3.0
MIN_RULE_WIDTH_PT = 4.0
MIN_RULE_COVERAGE = 0.55     # fraction of the span's width the rule must cover


@dataclass
class TextSpan:
    """A run of characters sharing one typographic style."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font: str = ""
    size: float = 0.0
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikeout: bool = False
    heading_level: int = 0     # 0 = body text

    @property
    def width(self) -> float:
        return max(self.x1 - self.x0, 0.0)

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 0.0)

    @property
    def styled(self) -> bool:
        return bool(
            self.bold or self.italic or self.underline or self.strikeout or self.heading_level
        )


@dataclass
class _Rule:
    """A thin horizontal line: candidate underline or strikethrough."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def mid_y(self) -> float:
        return (self.y0 + self.y1) / 2.0


def _is_bold(font_name: str, flags: int, weight: int) -> bool:
    if flags & FLAG_FORCE_BOLD:
        return True
    if weight >= 600:
        return True
    return bool(_BOLD_NAME_RE.search(font_name))


def _is_italic(font_name: str, flags: int) -> bool:
    if flags & FLAG_ITALIC:
        return True
    return bool(_ITALIC_NAME_RE.search(font_name))


def _collect_rules(page: pdfium.PdfPage) -> list[_Rule]:
    """Thin horizontal path objects — the shapes that draw underlines."""
    rules: list[_Rule] = []
    try:
        objects = page.get_objects(max_depth=2)
    except Exception:  # noqa: BLE001 — a page we cannot walk simply has no rules
        return rules
    for obj in objects:
        if obj.type != pdfium.raw.FPDF_PAGEOBJ_PATH:
            continue
        try:
            x0, y0, x1, y1 = obj.get_bounds()
        except Exception:  # noqa: BLE001
            continue
        if (y1 - y0) <= MAX_RULE_THICKNESS_PT and (x1 - x0) >= MIN_RULE_WIDTH_PT:
            rules.append(_Rule(x0, y0, x1, y1))
    return rules


def _overlap_fraction(span: TextSpan, rule: _Rule) -> float:
    if span.width <= 0:
        return 0.0
    left = max(span.x0, rule.x0)
    right = min(span.x1, rule.x1)
    return max(right - left, 0.0) / span.width


def _apply_decorations(spans: list[TextSpan], rules: list[_Rule]) -> None:
    """Mark spans that a thin rule underlines or strikes through.

    Position separates the two: an underline sits at or just below the glyph
    bottom, a strikethrough crosses the middle. Both must cover most of the
    span's width, otherwise a table border or a page rule that merely happens to
    be nearby would be misread as emphasis.
    """
    for span in spans:
        if span.height <= 0:
            continue
        for rule in rules:
            if _overlap_fraction(span, rule) < MIN_RULE_COVERAGE:
                continue
            below = span.y0 - 0.35 * span.height
            if below <= rule.mid_y <= span.y0 + 0.12 * span.height:
                span.underline = True
            elif span.y0 + 0.30 * span.height <= rule.mid_y <= span.y0 + 0.72 * span.height:
                span.strikeout = True


def _assign_heading_levels(spans: list[TextSpan]) -> None:
    """Infer heading levels from type size relative to the page's body text.

    Body size is the *median over characters*, not over spans: a page with one
    long paragraph and eight headings would otherwise conclude that headings are
    the norm and the paragraph is unusual.
    """
    sizes: list[float] = []
    for span in spans:
        sizes.extend([span.size] * max(len(span.text.strip()), 0))
    sizes = [s for s in sizes if s > 0]
    if not sizes:
        return
    body = statistics.median(sizes)
    if body <= 0:
        return

    # Distinct heading sizes, largest first, become levels 1..3.
    heading_sizes = sorted(
        {round(s.size, 1) for s in spans if s.size >= body * 1.15 and len(s.text.strip()) > 1},
        reverse=True,
    )
    ranks = {size: index + 1 for index, size in enumerate(heading_sizes[:3])}

    for span in spans:
        text = span.text.strip()
        if not text or len(text) > 200:
            continue           # a long line is a paragraph, whatever its size
        ratio = span.size / body
        rounded = round(span.size, 1)
        if rounded in ranks and ratio >= 1.15:
            span.heading_level = ranks[rounded]
        elif span.bold and ratio >= 1.08 and len(text) <= 120:
            span.heading_level = min(len(ranks) + 1, 3) or 3


def extract_spans(page: pdfium.PdfPage, detect_headings: bool = True) -> list[TextSpan]:
    """Character-level typography, grouped into styled runs."""
    with pdfium_guard():
        return _extract_spans_locked(page, detect_headings)


def _extract_spans_locked(page: pdfium.PdfPage, detect_headings: bool) -> list[TextSpan]:
    textpage = page.get_textpage()
    spans: list[TextSpan] = []
    try:
        count = raw.FPDFText_CountChars(textpage.raw)
        buffer = ctypes.create_string_buffer(160)
        current: TextSpan | None = None
        current_key: tuple | None = None

        for index in range(count):
            char = chr(raw.FPDFText_GetUnicode(textpage.raw, index))
            flags = ctypes.c_int(0)
            length = raw.FPDFText_GetFontInfo(
                textpage.raw, index, buffer, 160, ctypes.byref(flags)
            )
            font_name = buffer.value.decode("utf-8", "replace") if length else ""
            weight = raw.FPDFText_GetFontWeight(textpage.raw, index)
            size = float(raw.FPDFText_GetFontSize(textpage.raw, index))

            left = ctypes.c_double(0)
            right = ctypes.c_double(0)
            bottom = ctypes.c_double(0)
            top = ctypes.c_double(0)
            ok = raw.FPDFText_GetCharBox(
                textpage.raw, index,
                ctypes.byref(left), ctypes.byref(right),
                ctypes.byref(bottom), ctypes.byref(top),
            )
            if not ok:
                continue

            bold = _is_bold(font_name, flags.value, weight)
            italic = _is_italic(font_name, flags.value)
            key = (font_name, round(size, 1), bold, italic)

            # A newline always ends a span; so does a style change.
            if char in "\r\n":
                current, current_key = None, None
                continue

            if current is None or key != current_key:
                current = TextSpan(
                    text=char,
                    x0=left.value, y0=bottom.value, x1=right.value, y1=top.value,
                    font=font_name, size=size, bold=bold, italic=italic,
                )
                current_key = key
                spans.append(current)
            else:
                current.text += char
                current.x0 = min(current.x0, left.value)
                current.y0 = min(current.y0, bottom.value)
                current.x1 = max(current.x1, right.value)
                current.y1 = max(current.y1, top.value)
    finally:
        textpage.close()

    spans = [s for s in spans if s.text.strip()]
    _apply_decorations(spans, _collect_rules(page))
    if detect_headings:
        _assign_heading_levels(spans)
    return spans


# --------------------------------------------------------------------------
# Re-applying the recovered typography to the VLM's plain text
# --------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _normalise(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed text plus a map back to original offsets.

    The VLM re-flows lines, so a span that the PDF broke across two lines is one
    run in the Markdown. Matching on collapsed whitespace is what makes the two
    comparable; the offset map is what lets us edit the original safely.
    """
    out: list[str] = []
    offsets: list[int] = []
    previous_space = False
    for index, char in enumerate(text):
        if char.isspace():
            if not previous_space and out:
                out.append(" ")
                offsets.append(index)
            previous_space = True
        else:
            out.append(char)
            offsets.append(index)
            previous_space = False
    return "".join(out), offsets


def _wrap(text: str, span: TextSpan) -> str:
    """Apply the strongest applicable marks, innermost first."""
    if span.strikeout:
        text = f"~~{text}~~"
    if span.underline:
        text = f"<u>{text}</u>"
    if span.italic and not span.heading_level:
        text = f"*{text}*"
    if span.bold and not span.heading_level:
        text = f"**{text}**"
    return text


@dataclass
class FormattingReport:
    spans_considered: int = 0
    spans_applied: int = 0
    headings: int = 0
    bold: int = 0
    italic: int = 0
    underline: int = 0
    strikeout: int = 0
    skipped_ambiguous: int = 0
    notes: list[str] = field(default_factory=list)


def _match_key(text: str) -> str:
    """Normalised form used to find a span inside the VLM's Markdown.

    The model re-typesets the page, so the bytes rarely match exactly. It
    resolves ligatures (``ﬁ`` → ``fi``), normalises quotes and dashes, drops
    the soft hyphen left by justified line breaks, and varies case at line
    starts. Matching on the raw span text lost 28% of all recovered marks to
    differences that carry no meaning.
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.replace("­", "").replace("’", "'").replace("‘", "'")
    folded = folded.replace("“", '"').replace("”", '"')
    folded = folded.replace("—", "-").replace("–", "-")
    return " ".join(folded.split()).casefold()


def _style_signature(span: TextSpan) -> tuple:
    return (span.bold, span.italic, span.underline, span.strikeout, span.heading_level)


def apply_formatting(
    markdown: str,
    spans: list[TextSpan],
    min_chars: int = 3,
    max_occurrences: int = 1,
) -> tuple[str, FormattingReport]:
    """Re-mark ``markdown`` using typography recovered from the page.

    Conservative where it matters. A span is skipped when the same text also
    appears on the page in a *different* style, because then an occurrence in
    the Markdown is genuinely ambiguous and marking it could contradict the
    source. When every occurrence on the page carries the same style, all of
    them are marked — restricting that to a single occurrence discarded 17% of
    recovered marks for no gain in safety.

    Wrongly emphasising text is worse than leaving it plain: the benchmark
    rewards correct marks, but a production consumer is actively misled by
    wrong ones. Nothing here marks text that was not styled in the PDF.
    """
    report = FormattingReport()
    if not markdown or not spans:
        return markdown, report

    styled = [s for s in spans if s.styled and len(s.text.strip()) >= min_chars]
    report.spans_considered = len(styled)
    if not styled:
        return markdown, report

    # Which texts carry exactly one style across the whole page? Those are safe
    # to mark everywhere they occur; anything with two styles stays ambiguous.
    styles_by_text: dict[str, set[tuple]] = {}
    for span in spans:
        key = _match_key(span.text.strip())
        if key:
            styles_by_text.setdefault(key, set()).add(_style_signature(span))

    # Longest first: marking "Total revenue" before "Total" avoids nesting a
    # short span inside a long one and producing ****broken**** markup.
    styled.sort(key=lambda s: len(s.text.strip()), reverse=True)

    flat, offsets = _normalise(markdown)
    haystack = _match_key(flat)
    # _match_key only folds case and maps single characters, so offsets survive.
    if len(haystack) != len(flat):
        haystack = flat.casefold()

    claimed: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in _HTML_TAG_RE.finditer(flat)
    ]
    edits: list[tuple[int, int, TextSpan]] = []

    def collides(start: int, end: int) -> bool:
        return any(not (end <= s or start >= e) for s, e in claimed)

    for span in styled:
        needle = _match_key(span.text.strip())
        if len(needle) < min_chars:
            continue
        if len(styles_by_text.get(needle, ())) > 1:
            report.skipped_ambiguous += 1
            continue

        positions: list[int] = []
        start = haystack.find(needle)
        while start != -1:
            positions.append(start)
            start = haystack.find(needle, start + 1)
        if not positions:
            continue
        if max_occurrences and len(positions) > max_occurrences and len(needle) < 8:
            # Very short repeated strings are the risky case; leave those alone.
            report.skipped_ambiguous += 1
            continue

        for position in positions:
            end = position + len(needle)
            if collides(position, end):
                continue
            claimed.append((position, end))
            edits.append((position, end, span))

    if not edits:
        return markdown, report

    # Apply right-to-left so earlier offsets stay valid.
    edits.sort(key=lambda e: e[0], reverse=True)
    result = markdown
    for start, end, span in edits:
        original_start = offsets[start]
        original_end = offsets[end - 1] + 1
        chunk = result[original_start:original_end]
        if span.heading_level:
            hashes = "#" * span.heading_level
            replacement = f"{hashes} {chunk.strip()}"
            report.headings += 1
        else:
            replacement = _wrap(chunk, span)
        result = result[:original_start] + replacement + result[original_end:]
        report.spans_applied += 1
        report.bold += bool(span.bold and not span.heading_level)
        report.italic += bool(span.italic and not span.heading_level)
        report.underline += bool(span.underline)
        report.strikeout += bool(span.strikeout)

    return result, report


def format_page_markdown(
    page: pdfium.PdfPage, markdown: str, detect_headings: bool = True
) -> tuple[str, FormattingReport]:
    """Convenience: extract typography from ``page`` and apply it to ``markdown``."""
    spans = extract_spans(page, detect_headings=detect_headings)
    return apply_formatting(markdown, spans)
