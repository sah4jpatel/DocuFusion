"""Tier-2 client behaviour: retries, rotation, validity gates and fallback.

These paths only fire when the model misbehaves, which is precisely when they
matter and precisely when nobody exercises them by hand. The mock server's
``reply_queue`` scripts a sequence of replies so each failure mode is
reproducible.
"""

from __future__ import annotations

import pypdfium2 as pdfium
import pytest

from docfusion.config import VLMEndpoint
from docfusion.engines.olmocr_client import OlmOCRClient, OlmOCRError
from tests.conftest import olmocr_reply


@pytest.fixture()
def page(simple_pdf):
    pdf = pdfium.PdfDocument(str(simple_pdf))
    page = pdf[0]
    yield page
    page.close()
    pdf.close()


def client_for(mock_vllm, **overrides) -> OlmOCRClient:
    endpoint = VLMEndpoint(base_url=mock_vllm.base_url, **overrides)
    return OlmOCRClient(endpoint)


class TestHappyPath:
    def test_front_matter_stripped_and_flags_surfaced(self, page, mock_vllm):
        mock_vllm.markdown_reply = olmocr_reply(
            body="# Title\n\n<table><tr><td>1</td></tr></table>",
            primary_language="de", is_table=True,
        )
        result = client_for(mock_vllm).ocr_page(page, 0)
        assert result.markdown.startswith("# Title")
        assert "primary_language" not in result.markdown
        assert result.is_table is True
        assert result.primary_language == "de"
        assert result.attempts == 1
        assert not result.degraded

    def test_request_matches_the_trained_contract(self, page, mock_vllm):
        client_for(mock_vllm).ocr_page(page, 0)
        sent = mock_vllm.requests[-1]
        assert sent["max_tokens"] == 8000
        assert sent["temperature"] == 0.1
        assert [p["type"] for p in sent["messages"][0]["content"]] == ["text", "image_url"]
        # No anchored text layer: olmOCR-2 is a no-anchoring model.
        assert "RAW_TEXT_START" not in sent["messages"][0]["content"][0]["text"]

    def test_blank_page_is_a_valid_answer_not_a_retry(self, page, mock_vllm):
        mock_vllm.markdown_reply = olmocr_reply(body="", primary_language="null")
        result = client_for(mock_vllm).ocr_page(page, 0)
        assert result.markdown == ""
        assert result.attempts == 1
        assert len(mock_vllm.requests) == 1
        assert "no readable text" in result.note


class TestRetryLadder:
    def test_truncated_reply_is_retried_at_higher_temperature(self, page, mock_vllm):
        """finish_reason='length' means the page was cut off mid-generation."""
        mock_vllm.reply_queue = [(olmocr_reply(body="partial"), "length")]
        mock_vllm.markdown_reply = olmocr_reply(body="# Complete page")
        result = client_for(mock_vllm).ocr_page(page, 0)
        assert result.markdown == "# Complete page"
        assert result.attempts == 2
        temperatures = [r["temperature"] for r in mock_vllm.requests]
        assert temperatures == [0.1, 0.1]

    def test_degenerate_reply_is_retried(self, page, mock_vllm):
        mock_vllm.reply_queue = [(olmocr_reply(body="x " + "loop forever " * 40), "stop")]
        mock_vllm.markdown_reply = olmocr_reply(body="# Clean retry")
        result = client_for(mock_vllm).ocr_page(page, 0)
        assert result.markdown == "# Clean retry"
        assert result.attempts == 2

    def test_temperature_escalates_across_attempts(self, page, mock_vllm):
        mock_vllm.reply_queue = [
            (olmocr_reply(body="a"), "length"),
            (olmocr_reply(body="b"), "length"),
            (olmocr_reply(body="c"), "length"),
        ]
        mock_vllm.markdown_reply = olmocr_reply(body="# Finally")
        result = client_for(mock_vllm).ocr_page(page, 0)
        assert result.markdown == "# Finally"
        assert [r["temperature"] for r in mock_vllm.requests] == [0.1, 0.1, 0.2, 0.3]

    def test_exhausted_ladder_salvages_last_response(self, page, mock_vllm):
        mock_vllm.reply_queue = [(olmocr_reply(body=f"attempt {i}"), "length") for i in range(3)]
        mock_vllm.markdown_reply = olmocr_reply(body="still truncated")
        result = client_for(mock_vllm, max_attempts=3).ocr_page(page, 0)
        assert result.degraded
        assert result.attempts == 3
        assert result.markdown  # text preserved rather than discarded


class TestRotation:
    def test_sideways_page_is_re_rendered_and_retried(self, page, mock_vllm):
        mock_vllm.reply_queue = [
            (olmocr_reply(body="sideways garble", is_rotation_valid=False,
                          rotation_correction=90), "stop"),
        ]
        mock_vllm.markdown_reply = olmocr_reply(body="# Upright text")
        result = client_for(mock_vllm).ocr_page(page, 0)
        assert result.markdown == "# Upright text"
        assert result.rotation_applied == 90
        assert len(mock_vllm.requests) == 2
        # The retry carried a differently-sized image, i.e. it really re-rendered.
        first, second = (r["messages"][0]["content"][1]["image_url"]["url"]
                         for r in mock_vllm.requests)
        assert first != second


class TestFailureHandling:
    def test_unreachable_server_falls_back_to_text_layer(self, page):
        client = OlmOCRClient(VLMEndpoint(
            base_url="http://127.0.0.1:9/v1", max_attempts=1, timeout_s=2
        ))
        result = client.ocr_page(page, 0)
        assert result.fallback and result.degraded
        assert "Quarterly Business Review" in result.markdown  # embedded text layer
        assert result.warnings

    def test_fallback_can_be_disabled_for_strict_pipelines(self, page):
        client = OlmOCRClient(VLMEndpoint(
            base_url="http://127.0.0.1:9/v1", max_attempts=1, timeout_s=2,
            fallback_to_text_layer=False,
        ))
        with pytest.raises(OlmOCRError, match="all 1 attempts failed"):
            client.ocr_page(page, 0)


class TestPageBudget:
    """One page must not be able to hold a GPU slot indefinitely.

    Measured on ParseBench layout pages: median page latency 27s, p90 189s, and
    a single worst-case page consumed 105 minutes — starving the five other
    workers queued behind it. ``timeout_s`` caps one request; only a budget caps
    the ladder.
    """

    def test_budget_stops_the_ladder_and_keeps_the_best_answer(self, page, mock_vllm):
        # Always invalid, so without a budget this would run all 8 attempts.
        mock_vllm.reply_queue = [
            (olmocr_reply(body=f"attempt {i}"), "length") for i in range(8)
        ]
        mock_vllm.markdown_reply = olmocr_reply(body="still truncated")
        client = client_for(mock_vllm, page_budget_s=0.001)
        result = client.ocr_page(page, 0)

        assert result.attempts <= 2, "budget should cut the ladder short"
        assert result.markdown, "the best available answer must still be returned"
        assert result.degraded
        assert any("budget exhausted" in w for w in result.warnings)

    def test_generous_budget_does_not_interfere(self, page, mock_vllm):
        mock_vllm.reply_queue = [(olmocr_reply(body="partial"), "length")]
        mock_vllm.markdown_reply = olmocr_reply(body="# Complete page")
        result = client_for(mock_vllm, page_budget_s=600).ocr_page(page, 0)
        assert result.markdown == "# Complete page"
        assert not result.degraded

    def test_budget_never_prevents_the_first_attempt(self, page, mock_vllm):
        """Even a zero budget must try once; returning nothing is worse."""
        result = client_for(mock_vllm, page_budget_s=0.0).ocr_page(page, 0)
        assert result.markdown
        assert result.attempts == 1
