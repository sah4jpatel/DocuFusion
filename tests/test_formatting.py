"""Typography recovery from the PDF text layer.

olmOCR-2 emits plain text by construction — its prompt asks for "the plain text
representation" — so every emphasis, heading and strikethrough in the source is
lost. These tests pin the deterministic recovery that puts them back, including
the conservatism that keeps it from inventing emphasis that was not there.
"""

from __future__ import annotations

import pypdfium2 as pdfium
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from docfusion.config import PipelineConfig
from docfusion.formatting import (
    TextSpan,
    apply_formatting,
    extract_spans,
)
from docfusion.pipeline import DocFusionPipeline


@pytest.fixture(scope="module")
def styled_pdf(tmp_path_factory):
    """One page exercising every mark the module claims to recover."""
    path = tmp_path_factory.mktemp("fmt") / "styled.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, 740, "Quarterly Report")
    c.setFont("Helvetica", 11)
    c.drawString(72, 715, "Revenue grew in all regions this quarter.")
    c.setFont("Helvetica-Bold", 11)
    c.drawString(72, 700, "Payment is not required.")
    c.setFont("Helvetica-Oblique", 11)
    c.drawString(72, 685, "Figures are unaudited.")
    c.setFont("Helvetica", 11)
    c.drawString(72, 670, "See the appendix.")
    c.setLineWidth(0.6)
    c.line(72, 667.5, 155, 667.5)                    # underline: just below baseline
    c.setFont("Helvetica", 11)
    c.drawString(72, 650, "Old clause removed.")
    c.line(72, 653.5, 168, 653.5)                    # strikethrough: through the middle
    c.save()
    return path


@pytest.fixture(scope="module")
def spans(styled_pdf):
    pdf = pdfium.PdfDocument(str(styled_pdf))
    page = pdf[0]
    try:
        return extract_spans(page)
    finally:
        page.close()
        pdf.close()


def find(spans, needle: str) -> TextSpan:
    for span in spans:
        if needle in span.text:
            return span
    raise AssertionError(f"no span containing {needle!r}")


class TestExtraction:
    def test_bold_detected_from_font_name(self, spans):
        assert find(spans, "Payment is not required").bold

    def test_italic_detected_from_font_name(self, spans):
        assert find(spans, "Figures are unaudited").italic

    def test_body_text_is_not_marked(self, spans):
        body = find(spans, "Revenue grew in all regions")
        assert not body.styled

    def test_underline_detected_from_rule_below_baseline(self, spans):
        span = find(spans, "See the appendix")
        assert span.underline and not span.strikeout

    def test_strikeout_detected_from_rule_through_middle(self, spans):
        span = find(spans, "Old clause removed")
        assert span.strikeout and not span.underline

    def test_heading_detected_from_relative_size(self, spans):
        """Body size is the median over characters, so one big line is a heading."""
        assert find(spans, "Quarterly Report").heading_level == 1

    def test_spans_carry_coordinates(self, spans):
        span = find(spans, "Quarterly Report")
        assert span.x1 > span.x0 and span.y1 > span.y0


class TestApplication:
    def test_marks_are_reapplied_to_plain_vlm_text(self, spans):
        plain = (
            "Quarterly Report\n\nRevenue grew in all regions this quarter.\n"
            "Payment is not required.\nFigures are unaudited.\n"
            "See the appendix.\nOld clause removed."
        )
        out, report = apply_formatting(plain, spans)
        assert "# Quarterly Report" in out
        assert "**Payment is not required.**" in out
        assert "*Figures are unaudited.*" in out
        assert "<u>See the appendix.</u>" in out
        assert "~~Old clause removed.~~" in out
        assert "Revenue grew in all regions this quarter." in out   # untouched
        assert report.spans_applied == 5

    def test_unstyled_text_is_returned_unchanged(self, spans):
        plain = "Some sentence that never appears on the page."
        out, report = apply_formatting(plain, spans)
        assert out == plain
        assert report.spans_applied == 0

    def test_ambiguous_spans_are_skipped_not_guessed(self):
        """Marking the wrong occurrence is worse than marking none.

        A production consumer is actively misled by emphasis in the wrong place,
        so a span appearing twice is left alone rather than resolved by guess.
        """
        span = TextSpan(text="Total", x0=0, y0=0, x1=10, y1=10, bold=True, size=11)
        out, report = apply_formatting("Total revenue and Total cost", [span])
        assert out == "Total revenue and Total cost"
        assert report.skipped_ambiguous == 1

    def test_short_spans_are_ignored(self):
        span = TextSpan(text="of", x0=0, y0=0, x1=5, y1=10, bold=True, size=11)
        out, _ = apply_formatting("a lot of things", [span])
        assert out == "a lot of things"

    def test_html_tables_are_not_corrupted(self):
        """olmOCR-2 emits HTML tables; marks must never land inside a tag."""
        span = TextSpan(text="colspan", x0=0, y0=0, x1=40, y1=10, bold=True, size=11)
        table = '<table><tr><th colspan="2">Region</th></tr></table>'
        out, _ = apply_formatting(table, [span])
        assert out == table

    def test_reflowed_line_breaks_still_match(self):
        """The VLM re-flows lines, so matching collapses whitespace."""
        span = TextSpan(text="Payment is not required", x0=0, y0=0, x1=90, y1=10,
                        bold=True, size=11)
        out, report = apply_formatting("... Payment is\nnot required ...", [span])
        assert report.spans_applied == 1
        assert "**Payment is\nnot required**" in out

    def test_longest_span_wins_to_avoid_nesting(self):
        """Marking a short span inside a long one produces ****broken**** markup."""
        long_span = TextSpan(text="Total revenue", x0=0, y0=0, x1=80, y1=10,
                             bold=True, size=11)
        short_span = TextSpan(text="revenue", x0=0, y0=0, x1=40, y1=10,
                              italic=True, size=11)
        out, _ = apply_formatting("The Total revenue line", [long_span, short_span])
        assert out.count("**") == 2
        assert "****" not in out


class TestPipelineIntegration:
    def test_pipeline_applies_formatting(self, styled_pdf):
        cfg = PipelineConfig()
        cfg.tier2_enabled = False
        cfg.use_docling_tier1 = False
        result = DocFusionPipeline(cfg).convert(styled_pdf)
        assert result.formatting_marks_applied == 5
        assert "**Payment is not required.**" in result.markdown
        assert result.summary()["formatting_marks"] == 5

    def test_formatting_can_be_disabled(self, styled_pdf):
        cfg = PipelineConfig()
        cfg.tier2_enabled = False
        cfg.use_docling_tier1 = False
        cfg.recover_formatting = False
        result = DocFusionPipeline(cfg).convert(styled_pdf)
        assert result.formatting_marks_applied == 0
        assert "**" not in result.markdown

    def test_pages_with_existing_markup_are_left_alone(self, styled_pdf, monkeypatch):
        """Docling emits its own structure; re-marking would double it up."""
        import docfusion.pipeline as pl

        monkeypatch.setattr(
            pl, "tier1_extract",
            lambda *a, **k: {0: "# Already A Heading\n\n**already bold**"},
        )
        cfg = PipelineConfig()
        cfg.tier2_enabled = False
        result = DocFusionPipeline(cfg).convert(styled_pdf)
        assert result.formatting_marks_applied == 0
        assert result.markdown.count("**") == 2
