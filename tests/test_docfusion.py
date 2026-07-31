import json

import pypdfium2 as pdfium
import pytest
from openai import BadRequestError, OpenAI
from pydantic import BaseModel

from docfusion import PipelineConfig, audit
from docfusion.anchoring import anchored_prompt_for, ANCHOR_HEADER
from docfusion.engines.olmocr_client import (
    OlmOCRClient,
    detect_repetition_loop,
    truncate_repetition,
)
from docfusion.licenses import assert_compliant
from docfusion.pipeline import DocFusionPipeline
from docfusion.services.vllm_service import (
    extract_json,
    inject_schema_into_prompt,
    validate_against_schema,
)
from docfusion.triage.heuristics import Route, triage_pdf
from docfusion.cli import main as cli_main


# ---------------------------------------------------------------- licensing
class TestLicenses:
    def test_default_bom_is_compliant(self):
        res = audit()
        assert res.ok, res.violations
        assert_compliant()  # must not raise

    def test_restricted_weights_fail_audit(self):
        res = audit(["marker", "chandra"])
        assert not res.ok
        assert any("chandra" in v.lower() for v in res.violations)

    def test_surya_is_denylisted(self):
        res = audit(["surya"])
        assert not res.ok

    def test_unknown_component_fails_closed(self):
        res = audit(["mystery-model"])
        assert not res.ok


# ------------------------------------------------------------------ triage
class TestTriage:
    def test_simple_prose_routes_fast(self, simple_pdf):
        decisions = triage_pdf(simple_pdf)
        assert all(d.route is Route.FAST for d in decisions), [d.reasons for d in decisions]

    def test_math_dense_routes_vlm(self, math_pdf):
        (d,) = triage_pdf(math_pdf)
        assert d.route is Route.VLM
        assert any("math density" in r for r in d.reasons)

    def test_scan_routes_vlm(self, scan_pdf):
        (d,) = triage_pdf(scan_pdf)
        assert d.route is Route.VLM
        assert any("sparse text" in r or "image coverage" in r for r in d.reasons)

    def test_mixed_doc_routes_per_page(self, mixed_pdf):
        d0, d1 = triage_pdf(mixed_pdf)
        assert d0.route is Route.FAST
        assert d1.route is Route.VLM


# --------------------------------------------------------------- anchoring
class TestAnchoring:
    def test_born_digital_page_gets_anchors(self, simple_pdf):
        prompt = anchored_prompt_for(simple_pdf, 0)
        assert ANCHOR_HEADER in prompt
        assert "Quarterly Business Review" in prompt
        assert "[" in prompt and "]" in prompt  # coordinate markers

    def test_scanned_page_has_no_anchors(self, scan_pdf):
        prompt = anchored_prompt_for(scan_pdf, 0)
        assert ANCHOR_HEADER not in prompt  # pure visual OCR prompt

    def test_anchor_truncation_cap(self, simple_pdf):
        prompt = anchored_prompt_for(simple_pdf, 0, max_chars=300)
        assert len(prompt) < 1200
        assert "truncated" in prompt


# --------------------------------------------------- Marker→vLLM shim logic
class _Schema(BaseModel):
    corrected_markdown: str


class TestVLLMShim:
    def test_marker_still_uses_structured_outputs(self):
        """The shim exists only because Marker calls OpenAI's Structured Outputs.

        Pinned against the submodule: if Marker ever switches to a
        vLLM-compatible request shape, this fails and the shim can be deleted
        rather than carried forever as unexplained vendor code.
        """
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1]
                  / "third_party" / "marker" / "marker" / "services" / "openai.py")
        if not source.exists():
            pytest.skip("marker submodule not initialised")
        text = source.read_text(encoding="utf-8")
        assert "chat.completions.parse(" in text
        assert "response_format=response_schema" in text

    def test_schema_injected_into_prompt(self):
        out = inject_schema_into_prompt("Fix this table.", _Schema)
        assert "Fix this table." in out
        assert "JSON Schema" in out
        assert "corrected_markdown" in out

    @pytest.mark.parametrize("payload", [
        '{"corrected_markdown": "hi"}',
        '```json\n{"corrected_markdown": "hi"}\n```',
        'Sure! Here is the JSON:\n{"corrected_markdown": "hi"}\nHope that helps.',
    ])
    def test_tolerant_json_extraction(self, payload):
        data = validate_against_schema(extract_json(payload), _Schema)
        assert data == {"corrected_markdown": "hi"}

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            extract_json("no json here at all")

    def test_real_vllm_rejects_openai_structured_outputs(self, mock_vllm):
        """Documents the exact failure the shim exists to fix."""
        client = OpenAI(base_url=mock_vllm.base_url, api_key="x")
        with pytest.raises(BadRequestError, match="text.*or.*json_object"):
            client.chat.completions.create(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "s", "schema": {}}},
            )

    def test_shim_path_succeeds_where_stock_path_fails(self, mock_vllm):
        """The shim's request shape (json_object + schema-in-prompt) round-trips."""
        client = OpenAI(base_url=mock_vllm.base_url, api_key="x")
        resp = client.chat.completions.create(
            model="m",
            messages=[{"role": "user",
                       "content": [{"type": "text",
                                    "text": inject_schema_into_prompt("Fix table.", _Schema)}]}],
            response_format={"type": "json_object"},
            extra_body={"guided_json": _Schema.model_json_schema()},
        )
        data = validate_against_schema(extract_json(resp.choices[0].message.content), _Schema)
        assert "|" in data["corrected_markdown"]
        sent = mock_vllm.requests[-1]
        assert sent["response_format"] == {"type": "json_object"}
        assert sent["guided_json"]["properties"]["corrected_markdown"]


# ------------------------------------------------------- generation guards
class TestRepetitionGuards:
    def test_clean_text_passes(self):
        assert not detect_repetition_loop("A normal paragraph with varied words and content.")

    def test_phrase_loop_detected(self):
        looped = "Valid intro. " + "the same phrase " * 40
        assert detect_repetition_loop(looped)
        trimmed, degraded = truncate_repetition(looped)
        assert degraded
        assert len(trimmed) < len(looped)

    def test_whitespace_spiral_detected(self):
        assert detect_repetition_loop("Header text" + " " * 500)


# ------------------------------------------------------------- end-to-end
class TestPipeline:
    def _config(self, mock_vllm) -> PipelineConfig:
        cfg = PipelineConfig()
        cfg.vlm.base_url = mock_vllm.base_url
        cfg.use_docling_tier1 = False  # sandbox: no HF model downloads
        return cfg

    def test_mixed_document_end_to_end(self, mixed_pdf, mock_vllm):
        pipe = DocFusionPipeline(self._config(mock_vllm))
        result = pipe.convert(mixed_pdf)
        # page 0 came from the text layer, page 1 from the (mock) VLM
        assert result.tier2_pages == [1]
        assert 0 < result.tier2_fraction < 1
        assert "Plain narrative text" in result.markdown          # tier 1
        assert "Recovered Page" in result.markdown                # tier 2
        assert "\\(E = mc^2\\)" in result.markdown
        # The VLM request carried an image-bearing payload in the trained order.
        sent = mock_vllm.requests[-1]
        parts = sent["messages"][0]["content"]
        assert [p["type"] for p in parts] == ["text", "image_url"]

    def test_front_matter_never_leaks_into_output(self, mixed_pdf, mock_vllm):
        """The YAML header olmOCR-2 emits must be stripped, not passed through."""
        pipe = DocFusionPipeline(self._config(mock_vllm))
        result = pipe.convert(mixed_pdf)
        assert "primary_language" not in result.markdown
        assert "is_rotation_valid" not in result.markdown
        assert not result.markdown.lstrip().startswith("---")

    def test_simple_document_never_touches_vlm(self, simple_pdf, mock_vllm):
        pipe = DocFusionPipeline(self._config(mock_vllm))
        result = pipe.convert(simple_pdf)
        assert result.tier2_pages == []
        assert mock_vllm.requests == []  # zero GPU calls for clean prose
        assert "Quarterly Business Review" in result.markdown

    def test_repetition_guard_marks_degraded_pages(self, math_pdf, mock_vllm):
        mock_vllm.markdown_reply = "Equation list: " + "x^2 + y^2 = z^2 " * 60
        pipe = DocFusionPipeline(self._config(mock_vllm))
        result = pipe.convert(math_pdf)
        assert result.degraded_pages == [0]

    def test_license_audit_blocks_noncompliant_bom(self, monkeypatch):
        import docfusion.pipeline as pl
        monkeypatch.setattr(pl, "assert_compliant",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("License audit failed")))
        with pytest.raises(RuntimeError, match="License audit failed"):
            DocFusionPipeline(PipelineConfig())


# -------------------------------------------------------------------- CLI
class TestCLI:
    def test_audit_command(self, capsys):
        assert cli_main(["audit"]) == 0
        assert "AUDIT PASSED" in capsys.readouterr().out

    def test_triage_command(self, mixed_pdf, capsys):
        assert cli_main(["triage", str(mixed_pdf)]) == 0
        data = json.loads(capsys.readouterr().out)
        assert [d["route"] for d in data] == ["fast", "vlm"]


# --------------------------------------------------------- batch / tier1-only
class TestBatchAndTier1Only:
    def test_tier1_only_never_calls_vlm_but_reports_escalations(self, mixed_pdf, mock_vllm):
        cfg = PipelineConfig()
        cfg.vlm.base_url = mock_vllm.base_url
        cfg.use_docling_tier1 = False
        cfg.tier2_enabled = False
        result = DocFusionPipeline(cfg).convert(mixed_pdf)
        assert mock_vllm.requests == []          # VLM untouched
        assert result.tier2_pages == [1]         # ...but escalation candidates reported
        assert "Plain narrative text" in result.markdown

    def test_batch_cli(self, tmp_path, simple_pdf, mixed_pdf, mock_vllm, capsys):
        in_dir = tmp_path / "in"
        (in_dir / "nested").mkdir(parents=True)
        (in_dir / "a.pdf").write_bytes(simple_pdf.read_bytes())
        (in_dir / "nested" / "b.pdf").write_bytes(mixed_pdf.read_bytes())
        out_dir = tmp_path / "out"
        rc = cli_main(["batch", str(in_dir), str(out_dir),
                       "--vlm-base-url", mock_vllm.base_url, "--no-docling"])
        assert rc == 0
        assert (out_dir / "a.md").exists()
        assert (out_dir / "nested" / "b.md").exists()
        assert "Recovered Page" in (out_dir / "nested" / "b.md").read_text()
        err = capsys.readouterr().err
        assert "done: 2/2 converted" in err

    def test_batch_skip_existing(self, tmp_path, simple_pdf, mock_vllm, capsys):
        in_dir, out_dir = tmp_path / "in", tmp_path / "out"
        in_dir.mkdir(); out_dir.mkdir()
        (in_dir / "a.pdf").write_bytes(simple_pdf.read_bytes())
        (out_dir / "a.md").write_text("existing")
        rc = cli_main(["batch", str(in_dir), str(out_dir), "--skip-existing",
                       "--tier1-only", "--no-docling"])
        assert rc == 0
        assert (out_dir / "a.md").read_text() == "existing"

    def test_batch_continues_past_corrupt_pdf(self, tmp_path, simple_pdf, capsys):
        in_dir, out_dir = tmp_path / "in", tmp_path / "out"
        in_dir.mkdir()
        (in_dir / "bad.pdf").write_bytes(b"not a pdf at all")
        (in_dir / "good.pdf").write_bytes(simple_pdf.read_bytes())
        rc = cli_main(["batch", str(in_dir), str(out_dir), "--tier1-only", "--no-docling"])
        assert rc == 1                            # nonzero exit signals partial failure
        assert (out_dir / "good.md").exists()
        err = capsys.readouterr().err
        assert "FAIL bad.pdf" in err and "done: 1/2 converted" in err
