"""Input normalisation: images as documents, and undecodable text.

Both defects here were found by running ParseBench, not by reading the code —
they are the kind that never appear in a PDF-only test corpus and then take out
a production batch.
"""

from __future__ import annotations

import json

import pypdfium2 as pdfium
import pytest
from PIL import Image

from docfusion.config import PipelineConfig
from docfusion.io import IMAGE_SUFFIXES, as_pdf, image_to_pdf_bytes, is_image, sanitize_text
from docfusion.pipeline import DocFusionPipeline
from docfusion.triage.heuristics import triage_pdf


@pytest.fixture()
def png_scan(tmp_path):
    path = tmp_path / "scan.png"
    Image.new("RGB", (1200, 1600), (243, 241, 236)).save(path)
    return path


@pytest.fixture()
def transparent_png(tmp_path):
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (600, 800), (0, 0, 0, 0)).save(path)
    return path


class TestImageInput:
    def test_common_scan_formats_are_recognised(self):
        for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            assert suffix in IMAGE_SUFFIXES
        assert is_image("a.PNG") and is_image("b.jpeg")
        assert not is_image("c.pdf")

    def test_image_becomes_a_readable_single_page_pdf(self, png_scan):
        data = image_to_pdf_bytes(png_scan)
        pdf = pdfium.PdfDocument(data)
        try:
            assert len(pdf) == 1
        finally:
            pdf.close()

    def test_transparent_image_is_flattened_onto_white(self, transparent_png):
        """Left as alpha, a transparent scan renders as a black rectangle."""
        data = image_to_pdf_bytes(transparent_png)
        pdf = pdfium.PdfDocument(data)
        try:
            image = pdf[0].render(scale=0.25).to_pil().convert("RGB")
            assert image.getpixel((image.width // 2, image.height // 2)) == (255, 255, 255)
        finally:
            pdf.close()

    def test_as_pdf_passes_pdfs_through_untouched(self, simple_pdf):
        with as_pdf(simple_pdf) as path:
            assert path == simple_pdf

    def test_as_pdf_converts_images_and_cleans_up(self, png_scan):
        with as_pdf(png_scan) as path:
            assert path.suffix == ".pdf"
            assert path.exists()
            converted = path
        assert not converted.exists(), "temporary conversion should be removed"

    def test_triage_works_on_a_converted_image(self, png_scan):
        with as_pdf(png_scan) as path:
            decisions = triage_pdf(path)
        assert len(decisions) == 1

    def test_pipeline_accepts_an_image_directly(self, png_scan):
        """A .png used to fail with 'PDFium: Data format error'."""
        cfg = PipelineConfig()
        cfg.tier2_enabled = False
        cfg.use_docling_tier1 = False
        result = DocFusionPipeline(cfg).convert(png_scan)
        assert result.page_count == 1
        assert result.path == str(png_scan), "should report the caller's path, not the temp file"

    def test_batch_picks_up_images_alongside_pdfs(self, tmp_path, simple_pdf, png_scan):
        from docfusion.cli import main as cli_main

        in_dir, out_dir = tmp_path / "in", tmp_path / "out"
        in_dir.mkdir()
        (in_dir / "doc.pdf").write_bytes(simple_pdf.read_bytes())
        (in_dir / "scan.png").write_bytes(png_scan.read_bytes())

        rc = cli_main(["batch", str(in_dir), str(out_dir), "--tier1-only", "--no-docling"])
        assert rc == 0
        assert (out_dir / "doc.md").exists()
        assert (out_dir / "scan.md").exists()


class TestSurrogateHandling:
    def test_lone_surrogates_are_stripped(self):
        """They live happily in a str and raise only when something encodes."""
        text = "Total\ud800 revenue\udfff"
        with pytest.raises(UnicodeEncodeError):
            text.encode("utf-8")
        cleaned = sanitize_text(text)
        assert cleaned.encode("utf-8") == b"Total revenue"

    def test_clean_text_is_returned_unchanged(self):
        text = "Ordinary — text with ∑ symbols and é accents"
        assert sanitize_text(text) is text

    def test_empty_input_is_safe(self):
        assert sanitize_text("") == ""

    def test_sanitized_output_survives_json(self):
        """The real failure was 'Error serializing to JSON: surrogates not allowed'."""
        payload = {"markdown": sanitize_text("bad\ud800text")}
        assert json.dumps(payload)

    def test_pipeline_output_is_always_encodable(self, simple_pdf):
        cfg = PipelineConfig()
        cfg.tier2_enabled = False
        cfg.use_docling_tier1 = False
        result = DocFusionPipeline(cfg).convert(simple_pdf)
        result.markdown.encode("utf-8")
        json.dumps({"pages": result.page_markdown})
