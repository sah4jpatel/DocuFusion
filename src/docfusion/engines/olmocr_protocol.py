"""The olmOCR-2 wire contract, kept deliberately separate from transport.

olmOCR-2 (``allenai/olmOCR-2-7B-1025``) is a *fine-tuned* model: it only
performs at benchmark level when queried exactly the way it was trained. The
contract is defined by ``olmocr/pipeline.py`` upstream and reproduced here:

===========================  ====================================================
prompt                       ``build_no_anchoring_v4_yaml_prompt()`` — **no anchoring**
message order                text part FIRST, then the image part
image                        rendered so the *longest side* is 1288 px (not a DPI)
max_tokens                   8000
temperature                  escalating ladder per attempt, starting at 0.1
context ceiling              16384 total tokens; above that the page is invalid
validity                     ``finish_reason`` must be ``"stop"``
response                     YAML front matter + markdown body
tables                       HTML (``<table>``), not Markdown pipes
math                         ``\\( \\)`` inline, ``\\[ \\]`` block
figures                      ``![alt](page_startx_starty_width_height.png)``
rotation                     front matter may request a re-render and retry
===========================  ====================================================

Two details are easy to get wrong and cost real accuracy:

* **Anchoring is obsolete for this model.** olmOCR v1 injected the PDF text
  layer into the prompt; olmOCR-2 was trained *without* it, and upstream's own
  ``--target_anchor_text_len`` flag is documented "not used for new models".
  Sending anchors puts the model off-distribution and burns context. See
  :mod:`docfusion.anchoring`, which is retained for the fallback path only.
* **The reply is not bare Markdown.** It begins with a ``---`` YAML block. A
  client that returns ``response.choices[0].message.content`` verbatim leaks
  that front matter into every escalated page.

:data:`V4_YAML_PROMPT` is vendored rather than imported so the core package
stays installable without olmOCR's dependency tree. ``test_prompt_matches_
upstream`` compares it against the submodule and fails when upstream changes
it, so the pin is enforced rather than assumed.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any

import pypdfium2 as pdfium

from docfusion.pdfium_lock import pdfium_guard

# Verbatim from olmocr.prompts.prompts.build_no_anchoring_v4_yaml_prompt().
V4_YAML_PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values for the "
    "primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
)

# olmocr/pipeline.py TEMPERATURE_BY_ATTEMPT
TEMPERATURE_BY_ATTEMPT: tuple[float, ...] = (0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0)

# olmocr/pipeline.py build_page_query / try_single_page
MAX_OUTPUT_TOKENS = 8000
MODEL_MAX_CONTEXT = 16384
TARGET_LONGEST_IMAGE_DIM = 1288

# Constrains the reply to the front-matter shape (vLLM `guided_regex`).
FRONT_MATTER_GUIDED_REGEX = (
    r"---\nprimary_language: (?:[a-z]{2}|null)\n"
    r"is_rotation_valid: (?:True|False|true|false)\n"
    r"rotation_correction: (?:0|90|180|270)\n"
    r"is_table: (?:True|False|true|false)\n"
    r"is_diagram: (?:True|False|true|false)\n"
    r"(?:---|---\n[\s\S]+)"
)

VALID_ROTATIONS = (0, 90, 180, 270)


@dataclass
class PageResponse:
    """Parsed front matter plus the Markdown body (upstream ``PageResponse``)."""

    primary_language: str | None = None
    is_rotation_valid: bool = True
    rotation_correction: int = 0
    is_table: bool = False
    is_diagram: bool = False
    natural_text: str | None = None

    def __post_init__(self) -> None:
        if self.rotation_correction not in VALID_ROTATIONS:
            raise ValueError(f"rotation_correction must be one of {VALID_ROTATIONS}")


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return default


def _coerce_rotation(value: Any) -> int:
    try:
        rotation = int(value)
    except (TypeError, ValueError):
        return 0
    return rotation if rotation in VALID_ROTATIONS else 0


def split_front_matter(content: str) -> tuple[dict[str, Any], str]:
    """Split a ``---`` YAML front-matter block from the Markdown body.

    Mirrors upstream ``FrontMatterParser._extract_front_matter_and_text`` but
    degrades to a tolerant line parser when PyYAML is absent or the block is
    malformed — a partially understood header must never cost us the body text.
    """
    if not content:
        return {}, ""
    text = content.lstrip("﻿")
    if not text.startswith("---\n"):
        return {}, text.strip()

    end = text.find("\n---", 4)
    if end == -1:
        return {}, text.strip()

    block, body = text[4:end], text[end + 4:].strip()
    try:
        import yaml

        parsed = yaml.safe_load(block)
        if isinstance(parsed, dict):
            return parsed, body
    except Exception:  # noqa: BLE001 — fall through to the line parser
        pass

    parsed = {}
    for line in block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            parsed[key.strip()] = value.strip()
    return parsed, body


def parse_page_response(content: str) -> PageResponse:
    """Turn a raw olmOCR-2 completion into a :class:`PageResponse`.

    Missing or malformed keys fall back to permissive defaults; upstream raises
    instead, but for a batch pipeline dropping a page's text because its header
    lacked ``is_diagram`` is the worse failure.
    """
    front, body = split_front_matter(content)
    language = front.get("primary_language")
    if isinstance(language, str) and language.strip().lower() in {"null", "none", ""}:
        language = None
    return PageResponse(
        primary_language=language if isinstance(language, str) or language is None else None,
        is_rotation_valid=_coerce_bool(front.get("is_rotation_valid"), True),
        rotation_correction=_coerce_rotation(front.get("rotation_correction", 0)),
        is_table=_coerce_bool(front.get("is_table"), False),
        is_diagram=_coerce_bool(front.get("is_diagram"), False),
        natural_text=body or None,
    )


def render_page_png(
    page: pdfium.PdfPage,
    target_longest_dim: int = TARGET_LONGEST_IMAGE_DIM,
    rotation: int = 0,
) -> bytes:
    """Render a page so its longest side is ``target_longest_dim`` pixels.

    olmOCR-2 was trained on longest-side-normalised renders, so a fixed DPI
    would feed letter and A0 pages at wildly different effective resolutions.
    ``rotation`` is applied after rendering to honour ``rotation_correction``.

    The PDFium lock is taken *here* rather than left to callers. Relying on
    call-site discipline is how this crashed the first time: one unguarded
    caller is enough to corrupt the heap for every thread.
    """
    if rotation and rotation not in VALID_ROTATIONS:
        raise ValueError(f"rotation must be one of {VALID_ROTATIONS}")

    with pdfium_guard():
        width, height = page.get_width(), page.get_height()
        longest = max(width, height, 1.0)
        scale = target_longest_dim / longest
        pil = page.render(scale=scale).to_pil()

    if rotation:
        # PIL rotates counter-clockwise; the contract is clockwise degrees.
        pil = pil.rotate(-rotation, expand=True)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def build_messages(image_b64: str) -> list[dict[str, Any]]:
    """Build the chat payload — text part first, exactly as trained."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": V4_YAML_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ],
        }
    ]


# --------------------------------------------------------------------------
# Degenerate-generation guard
# --------------------------------------------------------------------------

@dataclass
class RepetitionReport:
    detected: bool = False
    kind: str = ""
    detail: str = ""
    ngram_size: int = 0
    repeats: int = 0
    spans: list[tuple[int, int]] = field(default_factory=list)


_WHITESPACE_SPIRAL = re.compile(r"[ \t]{200,}|(?:\r?\n){40,}")


# Upstream's RepeatDetector defaults to 10 because it runs per-token during
# streaming. Applied to a finished page, 10 only catches loops whose period is
# <= 10 characters; the common VLM failure is a repeated *phrase* or table row.
# An n-gram of size n only shows a back-to-back repeat when n is a multiple of
# the loop's period, so the window has to exceed realistic phrase length.
DEFAULT_REPETITION_NGRAM_SIZE = 64


def ngram_repeats(text: str, max_ngram_size: int = DEFAULT_REPETITION_NGRAM_SIZE) -> list[int]:
    """Trailing-repeat counts per n-gram size (upstream ``RepeatDetector``).

    ``result[n-1]`` is how many times the final ``n``-character n-gram repeats
    back-to-back at the end of the text.
    """
    result = [0] * max_ngram_size
    if not text:
        return result
    normalised = re.sub(r"\s+", " ", text)
    for size in range(1, max_ngram_size + 1):
        if len(normalised) < size:
            continue
        target = normalised[-size:]
        count, pos = 0, len(normalised) - size
        while pos >= 0 and normalised[pos:pos + size] == target:
            count += 1
            pos -= size
        result[size - 1] = count
    return result


# A runaway generation repeats until it hits max_tokens, so the repeated tail is
# enormous. Ordinary documents repeat short runs constantly — table-of-contents
# dot leaders, form underscores, columns of zeros — and those runs are short.
# Thresholding on the *length of the repeated span* separates the two; counting
# repeats alone does not, because "........." is nine repeats of a 1-gram.
DEFAULT_MIN_REPEAT_SPAN_CHARS = 200
MIN_REPEATS_FOR_LOOP = 3


def detect_degenerate(
    text: str,
    max_ngram_size: int = DEFAULT_REPETITION_NGRAM_SIZE,
    max_repeats: int = MIN_REPEATS_FOR_LOOP,
    min_span_chars: int = DEFAULT_MIN_REPEAT_SPAN_CHARS,
) -> RepetitionReport:
    """Flag repetition loops and whitespace spirals.

    Only *trailing* repetition counts: a loop the model never escaped. The span
    threshold is what makes this usable on real documents — an earlier
    count-only version fired on every page containing ``Chapter 1 .........``
    or ``Name: ________``, and since a flagged page is retried up to eight
    times, each false positive cost eight GPU passes and still ended up marked
    degraded.
    """
    if not text:
        return RepetitionReport()

    spiral = _WHITESPACE_SPIRAL.search(text)
    if spiral:
        return RepetitionReport(
            detected=True,
            kind="whitespace_spiral",
            detail=f"{spiral.end() - spiral.start()} consecutive whitespace chars",
            spans=[(spiral.start(), spiral.end())],
        )

    counts = ngram_repeats(text, max_ngram_size=max_ngram_size)
    best: tuple[int, int, int] | None = None  # (span, size, repeats)
    for size, repeats in enumerate(counts, start=1):
        if repeats < max(max_repeats, MIN_REPEATS_FOR_LOOP):
            continue
        span = size * repeats
        if span >= min_span_chars and (best is None or span > best[0]):
            best = (span, size, repeats)

    if best is None:
        return RepetitionReport()

    span, size, repeats = best
    return RepetitionReport(
        detected=True,
        kind="repetition_loop",
        detail=f"trailing {size}-gram repeats {repeats}x ({span} chars)",
        ngram_size=size,
        repeats=repeats,
    )


def truncate_degenerate(
    text: str,
    max_ngram_size: int = DEFAULT_REPETITION_NGRAM_SIZE,
    max_repeats: int = MIN_REPEATS_FOR_LOOP,
    min_span_chars: int = DEFAULT_MIN_REPEAT_SPAN_CHARS,
) -> tuple[str, RepetitionReport]:
    """Trim a degenerate tail, preserving one copy of the repeated unit."""
    report = detect_degenerate(text, max_ngram_size, max_repeats, min_span_chars)
    if not report.detected:
        return text, report

    if report.kind == "whitespace_spiral":
        cleaned = _WHITESPACE_SPIRAL.sub("\n\n", text)
        return cleaned.rstrip(), report

    size = report.ngram_size
    keep = len(text)
    unit = re.sub(r"\s+", " ", text)[-size:]
    # Walk back over the repeated tail in the original string.
    while keep - size >= 0 and re.sub(r"\s+", " ", text[keep - size:keep]) == unit:
        keep -= size
    return (text[: keep + size]).rstrip(), report
