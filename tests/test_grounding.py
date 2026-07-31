"""Layout reconstruction from the PDF text layer.

The two things worth pinning are the ones that are silently wrong rather than
loudly broken: the y-axis flip (PDF space is bottom-left, consumers expect
top-left, so an unflipped box is *plausible* and mirrored), and column
splitting (a two-column page read straight across yields boxes in an order no
consumer can follow).
"""

from __future__ import annotations

import pypdfium2 as pdfium
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from docfusion.config import PipelineConfig
from docfusion.grounding import document_layout, page_layout_from_text_layer
from docfusion.pipeline import DocFusionPipeline

PAGE_W, PAGE_H = letter


@pytest.fixture(scope="module")
def single_column_pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("layout") / "single.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, 740, "Annual Report")          # near the TOP of the page
    c.setFont("Helvetica", 11)
    for offset, line in enumerate([
        "Revenue increased across every region during the period.",
        "Operating costs declined for the third consecutive quarter.",
        "The board has approved the proposed dividend.",
    ]):
        c.drawString(72, 700 - offset * 15, line)
    c.save()
    return path


@pytest.fixture(scope="module")
def two_column_pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("layout") / "twocol.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica", 11)
    for row in range(8):
        y = 700 - row * 15
        c.drawString(60, y, f"Left column line number {row}")
        c.drawString(340, y, f"Right column line number {row}")
    c.save()
    return path


def layout_for(path):
    pdf = pdfium.PdfDocument(str(path))
    page = pdf[0]
    try:
        return page_layout_from_text_layer(page, 1)
    finally:
        page.close()
        pdf.close()


class TestCoordinates:
    def test_origin_is_top_left_not_pdf_bottom_left(self, single_column_pdf):
        """The heading is drawn near the top, so its y must be small.

        Skipping the flip yields a mirrored box that still looks like a valid
        box — this is the assertion that catches it.
        """
        layout = layout_for(single_column_pdf)
        heading = layout.blocks[0]
        assert heading.text.startswith("Annual Report")
        assert heading.y < 0.12, f"heading should be near the page top, got y={heading.y}"

    def test_boxes_are_normalised(self, single_column_pdf):
        layout = layout_for(single_column_pdf)
        assert layout.blocks
        for block in layout.blocks:
            assert 0.0 <= block.x <= 1.0
            assert 0.0 <= block.y <= 1.0
            assert 0.0 < block.w <= 1.0
            assert 0.0 < block.h <= 1.0

    def test_page_dimensions_are_reported(self, single_column_pdf):
        layout = layout_for(single_column_pdf)
        assert layout.page_number == 1
        assert layout.width == pytest.approx(PAGE_W, abs=1)
        assert layout.height == pytest.approx(PAGE_H, abs=1)


class TestBlocking:
    def test_heading_is_its_own_block(self, single_column_pdf):
        layout = layout_for(single_column_pdf)
        assert layout.blocks[0].label in {"Title", "Section"}
        assert "Revenue increased" not in layout.blocks[0].text

    def test_consecutive_body_lines_merge_into_one_paragraph(self, single_column_pdf):
        layout = layout_for(single_column_pdf)
        body = [b for b in layout.blocks if b.label == "Text"]
        assert len(body) == 1
        assert "Revenue increased" in body[0].text
        assert "dividend" in body[0].text

    def test_reading_order_is_dense_and_ascending(self, single_column_pdf):
        layout = layout_for(single_column_pdf)
        assert [b.reading_order for b in layout.blocks] == list(range(len(layout.blocks)))


class TestColumns:
    def test_columns_are_not_merged_across_the_gutter(self, two_column_pdf):
        """Grouping only by vertical centre merges the two columns into one row.

        The symptom is a full-width block containing both columns' text, and a
        reading order that runs across the page instead of down it.
        """
        layout = layout_for(two_column_pdf)
        assert layout.blocks, "expected blocks"
        for block in layout.blocks:
            assert not ("Left column" in block.text and "Right column" in block.text), (
                f"block spans the gutter: {block.text[:80]!r}"
            )

    def test_each_column_stays_narrow(self, two_column_pdf):
        layout = layout_for(two_column_pdf)
        for block in layout.blocks:
            assert block.w < 0.6, f"block is full-width, gutter not detected: w={block.w}"

    def test_left_column_is_read_before_right(self, two_column_pdf):
        layout = layout_for(two_column_pdf)
        left = [b.reading_order for b in layout.blocks if "Left column" in b.text]
        right = [b.reading_order for b in layout.blocks if "Right column" in b.text]
        assert left and right
        assert max(left) < min(right), "columns should be read down, not across"


class TestPipelineIntegration:
    def test_layout_is_off_by_default(self, single_column_pdf):
        cfg = PipelineConfig()
        cfg.tier2_enabled = False
        cfg.use_docling_tier1 = False
        result = DocFusionPipeline(cfg).convert(single_column_pdf)
        assert result.layout == []

    def test_layout_is_emitted_when_enabled(self, single_column_pdf):
        cfg = PipelineConfig()
        cfg.tier2_enabled = False
        cfg.use_docling_tier1 = False
        cfg.emit_layout = True
        result = DocFusionPipeline(cfg).convert(single_column_pdf)
        assert len(result.layout) == 1
        assert result.summary()["layout_blocks"] == len(result.layout[0].blocks)

    def test_layout_dict_shape_matches_consumer_schema(self, single_column_pdf):
        layout = layout_for(single_column_pdf)
        payload = layout.as_dict()
        assert payload["page_number"] == 1
        item = payload["items"][0]
        assert set(item) == {"type", "md", "value", "bbox"}
        assert set(item["bbox"]) == {"x", "y", "w", "h", "label"}


class TestScannedPages:
    def test_page_without_text_layer_yields_no_blocks(self, scan_pdf):
        """A scan has no glyphs to group; grounding must return empty, not crash."""
        layouts = document_layout(str(scan_pdf))
        assert len(layouts) == 1
        assert layouts[0].blocks == []
