"""Malformed and hostile inputs.

A batch pipeline meets these constantly: truncated uploads, password-protected
contracts, scanner output with a zero-byte page, files that are not PDFs at all.
The contract is deliberately split:

* **Per page**, defects are absorbed — one unreadable page must never cost the
  other 400 pages of a document, so it is escalated with a reason instead.
* **Per document**, defects raise — the caller (``docfusion batch``) catches,
  records the filename and continues, which is what makes the failure visible
  rather than silently producing an empty ``.md``.
"""

from __future__ import annotations

import pytest
from pypdfium2 import PdfiumError
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from docfusion.config import PipelineConfig
from docfusion.pipeline import DocFusionPipeline
from docfusion.triage.heuristics import Route, triage_pdf


@pytest.fixture()
def blank_pdf(tmp_path):
    """A page with no text and no images — a scanner artefact."""
    path = tmp_path / "blank.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.showPage()
    c.save()
    return path


@pytest.fixture()
def encrypted_pdf(tmp_path):
    path = tmp_path / "encrypted.pdf"
    c = canvas.Canvas(str(path), pagesize=letter, encrypt="s3cret")
    c.drawString(72, 720, "Confidential settlement terms.")
    c.save()
    return path


class TestMalformedDocuments:
    def test_not_a_pdf_raises_for_the_caller_to_record(self, tmp_path):
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"this is not a pdf at all")
        with pytest.raises(PdfiumError):
            triage_pdf(bad)

    def test_truncated_pdf_raises(self, tmp_path, simple_pdf):
        truncated = tmp_path / "truncated.pdf"
        truncated.write_bytes(simple_pdf.read_bytes()[:400])
        with pytest.raises(PdfiumError):
            triage_pdf(truncated)

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.pdf"
        empty.write_bytes(b"")
        with pytest.raises(PdfiumError):
            triage_pdf(empty)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            triage_pdf(tmp_path / "nope.pdf")


class TestEncryption:
    def test_encrypted_pdf_without_password_raises(self, encrypted_pdf):
        with pytest.raises(PdfiumError):
            triage_pdf(encrypted_pdf)

    def test_password_unlocks_the_document(self, encrypted_pdf):
        decisions = triage_pdf(encrypted_pdf, password="s3cret")
        assert len(decisions) == 1


class TestDegenerateButValidPages:
    def test_blank_page_escalates_rather_than_vanishing(self, blank_pdf):
        """No text and no images is indistinguishable from an unOCRed scan.

        Escalating is the safe read: a blank page costs one cheap VLM call,
        whereas routing it to Tier 1 silently emits nothing.
        """
        (decision,) = triage_pdf(blank_pdf)
        assert decision.route is Route.VLM
        assert any("sparse text" in r for r in decision.reasons)

    def test_blank_page_converts_without_error(self, blank_pdf, mock_vllm):
        cfg = PipelineConfig()
        cfg.vlm.base_url = mock_vllm.base_url
        cfg.use_docling_tier1 = False
        result = DocFusionPipeline(cfg).convert(blank_pdf)
        assert result.page_count == 1
        assert result.tier2_pages == [0]

    def test_tier1_only_on_blank_page_yields_empty_not_crash(self, blank_pdf):
        cfg = PipelineConfig()
        cfg.tier2_enabled = False
        cfg.use_docling_tier1 = False
        result = DocFusionPipeline(cfg).convert(blank_pdf)
        assert result.markdown == ""
        assert result.tier2_pages == [0]      # still reported as a would-be escalation


class TestBatchIsolation:
    def test_one_bad_document_does_not_stop_the_batch(self, tmp_path, simple_pdf):
        """The whole point of raising per-document: batch records and continues."""
        from docfusion.cli import main as cli_main

        in_dir, out_dir = tmp_path / "in", tmp_path / "out"
        in_dir.mkdir()
        (in_dir / "01_good.pdf").write_bytes(simple_pdf.read_bytes())
        (in_dir / "02_bad.pdf").write_bytes(b"garbage")
        (in_dir / "03_good.pdf").write_bytes(simple_pdf.read_bytes())

        rc = cli_main(["batch", str(in_dir), str(out_dir), "--tier1-only", "--no-docling"])
        assert rc == 1                                    # nonzero: something failed
        produced = sorted(p.name for p in out_dir.glob("*.md"))
        assert produced == ["01_good.md", "03_good.md"]   # the good ones still landed
        assert not (out_dir / "02_bad.md").exists()       # no empty file pretending to succeed
