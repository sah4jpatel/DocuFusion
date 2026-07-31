"""Page → Markdown client for a local vLLM instance serving olmOCR-2.

Used when docfusion escalates a page itself, without Marker in the loop. The
wire contract lives in :mod:`docfusion.engines.olmocr_protocol`; this module is
transport, retries and failure handling only.

Reliability behaviours, each mirroring upstream ``olmocr/pipeline.py``:

* **Temperature ladder.** A page that comes back invalid is retried at a higher
  temperature (0.1 → 1.0) rather than being failed outright. Greedy decoding is
  what *causes* repetition loops, so retrying at temperature 0 would reproduce
  the same degenerate output every time.
* **Rotation correction.** When the model reports the page is sideways it also
  reports the correction; the page is re-rendered rotated and re-sent.
* **Validity gates.** ``finish_reason != "stop"`` (truncation) and a total token
  count above the context ceiling both mark the attempt invalid.
* **Fallback.** After the ladder is exhausted the page degrades to its embedded
  text layer and is flagged, so a bad page costs recall on one page instead of
  aborting the document.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any

import pypdfium2 as pdfium

from docfusion.config import VLMEndpoint
from docfusion.pdfium_lock import pdfium_guard
from docfusion.engines.olmocr_protocol import (
    DEFAULT_REPETITION_NGRAM_SIZE,
    FRONT_MATTER_GUIDED_REGEX,
    MODEL_MAX_CONTEXT,
    TEMPERATURE_BY_ATTEMPT,
    PageResponse,
    build_messages,
    detect_degenerate,
    parse_page_response,
    render_page_png,
    truncate_degenerate,
)

logger = logging.getLogger(__name__)


@dataclass
class PageResult:
    page_index: int
    markdown: str
    degraded: bool = False
    note: str = ""
    attempts: int = 1
    rotation_applied: int = 0
    is_table: bool = False
    is_diagram: bool = False
    primary_language: str | None = None
    fallback: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    warnings: list[str] = field(default_factory=list)


class OlmOCRError(RuntimeError):
    """Raised when a page could not be OCRed and no fallback was permitted."""


def _page_text_layer(page: pdfium.PdfPage) -> str:
    with pdfium_guard():
        textpage = page.get_textpage()
        try:
            return (textpage.get_text_bounded() or "").strip()
        finally:
            textpage.close()


class OlmOCRClient:
    def __init__(self, endpoint: VLMEndpoint | None = None, client: Any | None = None):
        self.endpoint = endpoint or VLMEndpoint()
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.endpoint.base_url,
                api_key=self.endpoint.api_key,
                max_retries=0,  # the temperature ladder below is our retry policy
            )
        return self._client

    # -- single attempt ---------------------------------------------------
    def _attempt(
        self, page: pdfium.PdfPage, temperature: float, rotation: int
    ) -> tuple[PageResponse, bool, list[str], int, int]:
        """One request. Returns (response, is_valid, warnings, in_tok, out_tok).

        PDFium is serialised; the HTTP call deliberately is not. Rasterising a
        page costs tens of milliseconds against seconds of inference, so holding
        the lock across the request would throttle the whole pipeline to one
        page at a time for no reason.
        """
        png = render_page_png(   # self-guarding; holds the PDFium lock internally
            page,
            target_longest_dim=self.endpoint.target_longest_image_dim,
            rotation=rotation,
        )
        image_b64 = base64.b64encode(png).decode()

        extra_body: dict[str, Any] = {}
        if self.endpoint.use_guided_decoding:
            extra_body["guided_regex"] = FRONT_MATTER_GUIDED_REGEX

        completion = self.client.chat.completions.create(
            model=self.endpoint.model,
            messages=build_messages(image_b64),
            max_tokens=self.endpoint.max_output_tokens,
            temperature=temperature,
            timeout=self.endpoint.timeout_s,
            extra_body=extra_body or None,
        )

        warnings: list[str] = []
        is_valid = True
        choice = completion.choices[0]

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason not in (None, "stop"):
            is_valid = False
            warnings.append(f"finish_reason={finish_reason} (output truncated)")

        usage = getattr(completion, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        total = getattr(usage, "total_tokens", 0) or 0
        if total > MODEL_MAX_CONTEXT:
            is_valid = False
            warnings.append(f"total_tokens={total} exceeds context ceiling {MODEL_MAX_CONTEXT}")

        raw = choice.message.content or ""
        response = parse_page_response(raw)

        if response.natural_text and detect_degenerate(
            response.natural_text,
            self.endpoint.repetition_ngram_size,
            self.endpoint.repetition_max_repeats,
            self.endpoint.repetition_min_span_chars,
        ).detected:
            is_valid = False
            warnings.append("degenerate generation detected")

        return response, is_valid, warnings, in_tok, out_tok

    # -- public API -------------------------------------------------------
    def ocr_page(self, page: pdfium.PdfPage, page_index: int, **_legacy: Any) -> PageResult:
        """OCR one page, escalating temperature and correcting rotation as needed."""
        rotation = 0
        warnings: list[str] = []
        last: tuple[PageResponse, list[str], int, int] | None = None
        max_attempts = max(1, min(self.endpoint.max_attempts, len(TEMPERATURE_BY_ATTEMPT)))

        for attempt in range(max_attempts):
            temperature = TEMPERATURE_BY_ATTEMPT[attempt]
            try:
                response, is_valid, attempt_warnings, in_tok, out_tok = self._attempt(
                    page, temperature, rotation
                )
            except Exception as exc:  # noqa: BLE001 — transport/server errors are retryable
                warnings.append(f"attempt {attempt + 1} failed: {type(exc).__name__}: {exc}")
                logger.warning("page %d attempt %d failed: %s", page_index, attempt + 1, exc)
                continue

            warnings.extend(attempt_warnings)
            last = (response, attempt_warnings, in_tok, out_tok)

            # The model says the page is sideways — re-render and retry once per angle.
            if not response.is_rotation_valid and response.rotation_correction and rotation == 0:
                rotation = response.rotation_correction
                warnings.append(f"re-rendering rotated {rotation}°")
                continue

            if is_valid and response.natural_text:
                return PageResult(
                    page_index=page_index,
                    markdown=response.natural_text,
                    attempts=attempt + 1,
                    rotation_applied=rotation,
                    is_table=response.is_table,
                    is_diagram=response.is_diagram,
                    primary_language=response.primary_language,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    warnings=warnings,
                )

            # A genuinely blank page is a valid answer, not a failure to retry.
            if is_valid and not response.natural_text:
                return PageResult(
                    page_index=page_index,
                    markdown="",
                    attempts=attempt + 1,
                    rotation_applied=rotation,
                    note="model reported no readable text",
                    primary_language=response.primary_language,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    warnings=warnings,
                )

        # Ladder exhausted: salvage the best invalid answer, else the text layer.
        if last is not None and last[0].natural_text:
            salvaged, report = truncate_degenerate(
                last[0].natural_text,
                self.endpoint.repetition_ngram_size,
                self.endpoint.repetition_max_repeats,
                self.endpoint.repetition_min_span_chars,
            )
            return PageResult(
                page_index=page_index,
                markdown=salvaged,
                degraded=True,
                note=report.detail or "all attempts invalid; salvaged last response",
                attempts=max_attempts,
                rotation_applied=rotation,
                warnings=warnings,
                input_tokens=last[2],
                output_tokens=last[3],
            )

        if not self.endpoint.fallback_to_text_layer:
            raise OlmOCRError(
                f"page {page_index}: all {max_attempts} attempts failed: {'; '.join(warnings)}"
            )

        return PageResult(
            page_index=page_index,
            markdown=_page_text_layer(page),
            degraded=True,
            fallback=True,
            note="VLM unavailable or all attempts failed; fell back to embedded text layer",
            attempts=max_attempts,
            warnings=warnings,
        )


# Backwards-compatible aliases for the pre-protocol API.
def detect_repetition_loop(
    text: str, window: int = DEFAULT_REPETITION_NGRAM_SIZE, max_repeats: int = 8
) -> bool:
    return detect_degenerate(text, max_ngram_size=window, max_repeats=max_repeats).detected


def truncate_repetition(
    text: str, window: int = DEFAULT_REPETITION_NGRAM_SIZE, max_repeats: int = 8
) -> tuple[str, bool]:
    cleaned, report = truncate_degenerate(text, max_ngram_size=window, max_repeats=max_repeats)
    return cleaned, report.detected
