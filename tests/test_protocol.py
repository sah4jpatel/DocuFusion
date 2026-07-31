"""Tests for the olmOCR-2 wire contract.

The suite that shipped before this module passed while the client sent a
hand-written prompt, put the image before the text, capped output at 4096
tokens and returned the model's YAML front matter verbatim — because the mock
returned whatever the client expected. These tests pin the contract to
upstream instead of to our own code.
"""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
import pytest

from docfusion.engines.olmocr_protocol import (
    MAX_OUTPUT_TOKENS,
    MODEL_MAX_CONTEXT,
    TARGET_LONGEST_IMAGE_DIM,
    TEMPERATURE_BY_ATTEMPT,
    V4_YAML_PROMPT,
    build_messages,
    detect_degenerate,
    ngram_repeats,
    parse_page_response,
    render_page_png,
    split_front_matter,
    truncate_degenerate,
)

OLMOCR_REPO = Path(__file__).resolve().parents[1] / "third_party" / "olmocr"


class TestUpstreamPin:
    """Fail loudly when the vendored submodule moves out from under us."""

    @pytest.mark.skipif(not OLMOCR_REPO.exists(), reason="olmOCR submodule not initialised")
    def test_prompt_matches_upstream(self):
        import sys

        sys.path.insert(0, str(OLMOCR_REPO))
        try:
            from olmocr.prompts.prompts import build_no_anchoring_v4_yaml_prompt
        except ImportError:
            pytest.skip("olmOCR prompts module not importable")
        finally:
            sys.path.remove(str(OLMOCR_REPO))

        def normalise(s: str) -> str:
            return " ".join(s.split())

        assert normalise(V4_YAML_PROMPT) == normalise(build_no_anchoring_v4_yaml_prompt()), (
            "The vendored olmOCR-2 prompt drifted from upstream. Update "
            "docfusion.engines.olmocr_protocol.V4_YAML_PROMPT to match, or the model "
            "runs off-distribution and benchmark scores drop."
        )

    def test_inference_constants_match_upstream(self):
        # olmocr/pipeline.py: MAX_TOKENS, MODEL_MAX_CONTEXT, --target_longest_image_dim
        assert MAX_OUTPUT_TOKENS == 8000
        assert MODEL_MAX_CONTEXT == 16384
        assert TARGET_LONGEST_IMAGE_DIM == 1288
        assert TEMPERATURE_BY_ATTEMPT[0] == 0.1, "greedy decoding causes the repetition loops"
        assert TEMPERATURE_BY_ATTEMPT == (0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0)

    def test_prompt_declares_no_anchoring(self):
        """olmOCR-2 is a no-anchoring model; the prompt must carry no text layer."""
        assert "RAW_TEXT_START" not in V4_YAML_PROMPT
        assert "front matter" in V4_YAML_PROMPT


class TestMessageShape:
    def test_text_precedes_image(self):
        """Trained order is text-then-image; flipping it changes the distribution."""
        messages = build_messages("Zm9v")
        parts = messages[0]["content"]
        assert [p["type"] for p in parts] == ["text", "image_url"]
        assert parts[0]["text"] == V4_YAML_PROMPT
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


class TestFrontMatter:
    def test_splits_yaml_header_from_body(self):
        raw = ("---\nprimary_language: en\nis_rotation_valid: True\n"
               "rotation_correction: 0\nis_table: False\nis_diagram: False\n---\n"
               "# Heading\n\nBody text.")
        front, body = split_front_matter(raw)
        assert front["primary_language"] == "en"
        assert body.startswith("# Heading")
        assert "primary_language" not in body

    def test_parsed_response_exposes_flags(self):
        raw = ("---\nprimary_language: fr\nis_rotation_valid: False\n"
               "rotation_correction: 90\nis_table: True\nis_diagram: False\n---\nLe texte.")
        response = parse_page_response(raw)
        assert response.primary_language == "fr"
        assert response.is_rotation_valid is False
        assert response.rotation_correction == 90
        assert response.is_table is True
        assert response.natural_text == "Le texte."

    def test_bare_markdown_survives(self):
        """A reply with no front matter must not lose its body."""
        response = parse_page_response("Just some text with no header.")
        assert response.natural_text == "Just some text with no header."

    def test_malformed_front_matter_keeps_body(self):
        raw = "---\nnot: [valid: yaml\n---\nThe body survives."
        response = parse_page_response(raw)
        assert response.natural_text == "The body survives."

    def test_null_language_becomes_none(self):
        raw = ("---\nprimary_language: null\nis_rotation_valid: True\nrotation_correction: 0\n"
               "is_table: False\nis_diagram: False\n---\n")
        assert parse_page_response(raw).primary_language is None

    def test_invalid_rotation_falls_back_to_zero(self):
        raw = ("---\nprimary_language: en\nis_rotation_valid: True\nrotation_correction: 45\n"
               "is_table: False\nis_diagram: False\n---\nx")
        assert parse_page_response(raw).rotation_correction == 0

    def test_blank_page_yields_no_text(self):
        raw = ("---\nprimary_language: null\nis_rotation_valid: True\nrotation_correction: 0\n"
               "is_table: False\nis_diagram: False\n---\n")
        assert parse_page_response(raw).natural_text is None


class TestRendering:
    def test_longest_side_is_normalised(self, simple_pdf):
        import io

        from PIL import Image

        pdf = pdfium.PdfDocument(str(simple_pdf))
        try:
            page = pdf[0]
            png = render_page_png(page, target_longest_dim=1288)
            page.close()
        finally:
            pdf.close()
        img = Image.open(io.BytesIO(png))
        assert max(img.size) == pytest.approx(1288, abs=2)

    def test_rotation_swaps_dimensions(self, simple_pdf):
        import io

        from PIL import Image

        pdf = pdfium.PdfDocument(str(simple_pdf))
        try:
            page = pdf[0]
            upright = Image.open(io.BytesIO(render_page_png(page, 1288, rotation=0))).size
            turned = Image.open(io.BytesIO(render_page_png(page, 1288, rotation=90))).size
            page.close()
        finally:
            pdf.close()
        assert turned == (upright[1], upright[0])


class TestDegenerateGuards:
    def test_clean_prose_passes(self):
        assert not detect_degenerate("A normal paragraph with varied words and content.").detected

    def test_trailing_loop_detected(self):
        looped = "Valid intro. " + "the same phrase " * 40
        report = detect_degenerate(looped)
        assert report.detected and report.kind == "repetition_loop"
        trimmed, _ = truncate_degenerate(looped)
        assert len(trimmed) < len(looped)
        assert "Valid intro." in trimmed

    def test_whitespace_spiral_detected(self):
        report = detect_degenerate("Header text" + " " * 500)
        assert report.detected and report.kind == "whitespace_spiral"

    @pytest.mark.parametrize("label,text", [
        ("toc dot leaders", "Chapter 1 " + "." * 40),
        ("form underscores", "Name: " + "_" * 30),
        ("column of zeros", "Revenue " + "0" * 20),
        ("rule of dashes", "Section " + "-" * 50),
    ])
    def test_ordinary_document_runs_are_not_loops(self, label, text):
        """Counting repeats alone flags every table of contents.

        These fired under a count-only threshold, and because a flagged page is
        retried the full length of the temperature ladder, each false positive
        cost eight GPU passes and still came back marked degraded.
        """
        assert not detect_degenerate(text).detected, label

    def test_long_runaway_span_is_still_caught(self):
        assert detect_degenerate("intro " + "a" * 500).detected
        assert detect_degenerate("intro " + "the same phrase " * 40).detected

    def test_markdown_table_is_not_degenerate(self):
        """Real tables repeat separators constantly — that is not a loop.

        The previous regex-based guard scanned the whole body, so any table with
        enough rows tripped it and got truncated as 'degraded'.
        """
        table = "| Region | Q1 | Q2 |\n|---|---|---|\n" + "".join(
            f"| Region {i} | {i * 10} | {i * 20} |\n" for i in range(60)
        )
        assert not detect_degenerate(table).detected

    def test_ngram_counter_matches_upstream_semantics(self):
        # olmocr RepeatDetector: 'abab' → 1 repeat of 'b', 2 of 'ab', 1 of 'bab'
        assert ngram_repeats("abab", max_ngram_size=3) == [1, 2, 1]
