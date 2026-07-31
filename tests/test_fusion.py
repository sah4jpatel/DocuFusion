"""Region routing and specialist dispatch.

The router's job is to find the parts of a page the generalist handles badly and
send them somewhere better. Two failure modes matter more than raw recall:

* routing a **ruled table** to a chart model — wasted GPU on something Tier 1
  already does well, and a worse answer;
* removing the generalist's text when a specialist runs — fusion must only ever
  add information.
"""

from __future__ import annotations

import pypdfium2 as pdfium
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from docfusion.fusion import (
    DetectedRegion,
    crop_region,
    detect_regions,
    fuse_page,
)
from docfusion.specialists.base import (
    Region,
    RegionKind,
    SpecialistResult,
    available_specialists,
    registry_bom,
)
from docfusion.specialists.charts import linearised_to_markdown
from docfusion.specialists.formulas import wrap_latex


@pytest.fixture(scope="module")
def chart_pdf(tmp_path_factory):
    """Bars and gridlines: many paths, almost no text over them."""
    path = tmp_path_factory.mktemp("fusion") / "chart.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica", 11)
    c.drawString(72, 740, "Regional revenue for the fourth quarter is shown below.")
    for i in range(14):
        c.rect(90 + i * 22, 480, 16, 40 + i * 9, fill=1)
    for g in range(10):
        c.line(85, 480 + g * 18, 400, 480 + g * 18)
    c.save()
    return path


@pytest.fixture(scope="module")
def table_pdf(tmp_path_factory):
    """A ruled table: also many paths, but dense with characters."""
    path = tmp_path_factory.mktemp("fusion") / "table.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica", 9)
    rows, cols = 12, 5
    x0, y0, cw, ch = 80, 400, 88, 20
    for r in range(rows + 1):
        c.line(x0, y0 + r * ch, x0 + cols * cw, y0 + r * ch)
    for col in range(cols + 1):
        c.line(x0 + col * cw, y0, x0 + col * cw, y0 + rows * ch)
    for r in range(rows):
        for col in range(cols):
            c.drawString(x0 + col * cw + 4, y0 + r * ch + 6, f"Cell {r}-{col} value")
    c.save()
    return path


def regions_of(path):
    pdf = pdfium.PdfDocument(str(path))
    page = pdf[0]
    try:
        return detect_regions(page), page, pdf
    except Exception:
        page.close()
        pdf.close()
        raise


class TestRegionDetection:
    def test_chart_cluster_is_detected(self, chart_pdf):
        regions, page, pdf = regions_of(chart_pdf)
        try:
            charts = [r for r in regions if r.kind is RegionKind.CHART]
            assert len(charts) == 1
            assert charts[0].path_count >= 12
            assert charts[0].area > 0.05
        finally:
            page.close()
            pdf.close()

    def test_ruled_table_is_not_routed_to_a_chart_model(self, table_pdf):
        """Path count alone would flag this; text density is what saves it."""
        regions, page, pdf = regions_of(table_pdf)
        try:
            assert [r for r in regions if r.kind is RegionKind.CHART] == []
        finally:
            page.close()
            pdf.close()

    def test_text_only_page_yields_no_regions(self, simple_pdf):
        regions, page, pdf = regions_of(simple_pdf)
        try:
            assert regions == []
        finally:
            page.close()
            pdf.close()

    def test_scanned_page_image_is_a_figure(self, scan_pdf):
        regions, page, pdf = regions_of(scan_pdf)
        try:
            assert any(r.kind is RegionKind.FIGURE for r in regions)
        finally:
            page.close()
            pdf.close()

    def test_crop_is_padded_and_nonempty(self, chart_pdf):
        regions, page, pdf = regions_of(chart_pdf)
        try:
            image = crop_region(page, regions[0])
            assert image is not None
            assert min(image.size) > 20
        finally:
            page.close()
            pdf.close()


class TestFusionSplicing:
    def test_specialist_output_is_appended_not_substituted(self, chart_pdf, monkeypatch):
        """Fusion may add information; it must never delete the generalist's."""
        import docfusion.fusion as fusion_module

        class FakeChartModel:
            name = "fake"
            kinds = (RegionKind.CHART, RegionKind.FIGURE)
            licence = "Apache-2.0"
            origin = "test"

            def available(self) -> bool:
                return True

            def run(self, region: Region) -> SpecialistResult:
                return SpecialistResult(markdown="| a | b |\n|---|---|\n| 1 | 2 |",
                                        specialist=self.name)

        monkeypatch.setattr(
            fusion_module, "available_specialists",
            lambda kind=None: {"fake": FakeChartModel()},
        )
        pdf = pdfium.PdfDocument(str(chart_pdf))
        page = pdf[0]
        try:
            original = "Regional revenue for the fourth quarter is shown below."
            result = fuse_page(page, 0, original)
        finally:
            page.close()
            pdf.close()

        assert original in result.markdown          # generalist text preserved
        assert "| a | b |" in result.markdown       # specialist output added
        assert result.report.by_specialist == {"fake": 1}

    def test_missing_specialist_falls_back_without_loss(self, chart_pdf, monkeypatch):
        import docfusion.fusion as fusion_module

        monkeypatch.setattr(fusion_module, "available_specialists", lambda kind=None: {})
        pdf = pdfium.PdfDocument(str(chart_pdf))
        page = pdf[0]
        try:
            result = fuse_page(page, 0, "original text")
        finally:
            page.close()
            pdf.close()
        assert result.markdown == "original text"
        assert result.report.fell_back >= 1

    def test_page_without_regions_is_untouched(self, simple_pdf):
        pdf = pdfium.PdfDocument(str(simple_pdf))
        page = pdf[0]
        try:
            result = fuse_page(page, 0, "unchanged")
        finally:
            page.close()
            pdf.close()
        assert result.markdown == "unchanged"
        assert result.report.regions == 0


class TestChartConversion:
    def test_deplot_output_becomes_a_markdown_table(self):
        """Values must end up adjacent to both their labels.

        The chart tests ask "series X at category Y equals Z", so flattening to
        prose would separate the number from what identifies it.
        """
        raw = ("TITLE | E-Government <0x0A> Category | IF | CP <0x0A> "
               "193 UN Member States | 0.8079 | 0.6653")
        md = linearised_to_markdown(raw)
        assert "**E-Government**" in md
        assert "| Category | IF | CP |" in md
        assert "| 193 UN Member States | 0.8079 | 0.6653 |" in md

    def test_ragged_rows_are_padded(self):
        md = linearised_to_markdown("a | b | c <0x0A> 1 | 2")
        assert md.count("|") % 2 == 0
        assert "| 1 | 2 |  |" in md

    def test_empty_output_is_empty_not_a_broken_table(self):
        assert linearised_to_markdown("") == ""
        assert linearised_to_markdown("   ") == ""


class TestFormulaWrapping:
    def test_uses_the_same_delimiters_as_the_generalist(self):
        """olmOCR-2 emits \\( \\) and \\[ \\]; mixing in $...$ would be incoherent."""
        assert wrap_latex("x^2", display=False) == "\\(x^2\\)"
        assert wrap_latex("x^2", display=True) == "\\[x^2\\]"

    def test_already_wrapped_latex_is_left_alone(self):
        assert wrap_latex("\\(a+b\\)") == "\\(a+b\\)"

    def test_empty_stays_empty(self):
        assert wrap_latex("") == ""


class TestRegistryLicensing:
    def test_every_specialist_declares_a_permissive_licence(self):
        """A specialist cannot enter the pipeline without its licence visible."""
        import docfusion.specialists.charts  # noqa: F401
        import docfusion.specialists.formulas  # noqa: F401

        bom = registry_bom()
        assert bom, "no specialists registered"
        for row in bom:
            assert row["licence"] in {"Apache-2.0", "MIT"}, row
            assert row["origin"]

    def test_uninstalled_specialists_are_simply_absent(self):
        """Optional dependencies must degrade, never raise."""
        found = available_specialists(RegionKind.CHART)
        assert isinstance(found, dict)

    def test_region_area_and_density_math(self):
        region = DetectedRegion(kind=RegionKind.CHART, x=0.1, y=0.1, w=0.5, h=0.2,
                                text_chars=60)
        assert region.area == pytest.approx(0.1)
        assert region.text_density == pytest.approx(600.0)
