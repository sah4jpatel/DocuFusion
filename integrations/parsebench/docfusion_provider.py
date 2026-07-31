"""ParseBench provider for DocFusion.

ParseBench (LlamaIndex) scores document parsers on five capability dimensions:
tables, charts, content faithfulness, semantic formatting and visual grounding.
Scoring is fully deterministic — no LLM judge — so it runs offline against a
local vLLM.

This file is installed into a ParseBench checkout by ``install.py``; it lives
here rather than in a ParseBench fork so the adapter is versioned with the
pipeline it adapts.

Two dimensions deserve honesty up front:

* **Visual grounding** asks a parser to trace every extracted element back to a
  bounding box on the page. olmOCR-2 emits linearised Markdown and no
  coordinates, so this provider cannot answer those tests. It reports layout
  only when Tier 1 (Docling) produced it. Expect a floor score, as every other
  Markdown-only entry on the leaderboard gets.
* **Charts** asks for exact data points with series and axis labels. olmOCR-2 is
  trained to emit a figure *placeholder*
  (``![alt](page_startx_starty_width_height.png)``), not chart series. This is
  an architectural limit of the model, not a tuning problem.

Both are reported rather than worked around: a benchmark you have quietly
routed around is not a measurement.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import PageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

_PIPELINE_CACHE: dict[tuple, Any] = {}


def _build_pipeline(config: dict[str, Any]):
    """One DocFusionPipeline per distinct config, reused across pages.

    Construction runs the license audit and (for hybrid) loads Docling's
    models, so rebuilding per page would dominate the run.
    """
    from docfusion.config import PipelineConfig
    from docfusion.pipeline import DocFusionPipeline

    mode = config.get("mode", "vlm_only")
    base_url = config.get("base_url") or os.getenv(
        "DOCFUSION_SERVER_URL", "http://localhost:8000/v1"
    )
    model = config.get("model") or os.getenv(
        "DOCFUSION_MODEL", "allenai/olmOCR-2-7B-1025"
    )
    key = (mode, base_url, model, config.get("workers"))
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]

    cfg = PipelineConfig()
    cfg.vlm.base_url = base_url
    cfg.vlm.model = model
    cfg.max_tier2_workers = int(config.get("workers") or 1)

    if mode == "vlm_only":
        cfg.force_tier2_all = True
        cfg.use_docling_tier1 = False       # nothing routes to Tier 1
    elif mode == "hybrid":
        cfg.force_tier2_all = False
        cfg.use_docling_tier1 = bool(config.get("docling", True))
    elif mode == "tier1_only":
        cfg.tier2_enabled = False
        cfg.use_docling_tier1 = bool(config.get("docling", True))
    else:
        raise ProviderConfigError(
            f"unknown docfusion mode {mode!r}; expected vlm_only | hybrid | tier1_only"
        )

    pipeline = DocFusionPipeline(cfg)
    _PIPELINE_CACHE[key] = pipeline
    return pipeline


@register_provider("docfusion")
class DocFusionProvider(Provider):
    """DocFusion: heuristic triage + Docling Tier 1 + olmOCR-2 Tier 2 (all Apache-2.0/MIT)."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"DocFusionProvider only supports PARSE, got {request.product_type}"
            )

        pdf_path = Path(request.source_file_path)
        if not pdf_path.exists():
            raise ProviderPermanentError(f"File not found: {pdf_path}")

        config = {**(self.base_config or {}), **(pipeline.config or {})}
        started_at = datetime.now()
        try:
            df = _build_pipeline(config)
            result = df.convert(pdf_path)
        except ProviderConfigError:
            raise
        except Exception as e:  # noqa: BLE001 — surfaced to ParseBench as a page failure
            raise ProviderPermanentError(f"DocFusion error on {pdf_path.name}: {e}") from e

        completed_at = datetime.now()
        pages = [
            {"page_index": index, "text": result.page_markdown.get(index, "")}
            for index in sorted(result.page_markdown)
        ]
        raw_output: dict[str, Any] = {
            "pages": pages or [{"page_index": 0, "text": result.markdown}],
            "text": result.markdown,
            "num_pages": len(result.decisions),
            # Kept for analysis: which pages cost GPU, and which degraded.
            "docfusion": {
                "tier2_pages": result.tier2_pages,
                "degraded_pages": result.degraded_pages,
                "fallback_pages": result.fallback_pages,
                "escalation_rate": result.tier2_fraction,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
        }
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        pages: list[PageIR] = []
        page_texts: list[str] = []
        for page_data in raw_result.raw_output.get("pages", []):
            # olmOCR-2's v4 prompt already emits HTML tables, which is what the
            # table metric parses. No pipe-table conversion needed — and doing
            # one would corrupt the HTML the model already produced.
            text = page_data.get("text", "") or ""
            pages.append(PageIR(page_index=page_data.get("page_index", 0), markdown=text))
            page_texts.append(text)

        full_text = raw_result.raw_output.get("text") or "\n\n".join(page_texts)
        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            markdown=full_text,
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
