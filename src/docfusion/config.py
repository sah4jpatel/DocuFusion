"""Central configuration for the dual-tier pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field

from docfusion.engines.olmocr_protocol import (
    MAX_OUTPUT_TOKENS,
    TARGET_LONGEST_IMAGE_DIM,
    TEMPERATURE_BY_ATTEMPT,
)

# Ampere (8.0/8.6 — A100, RTX 3090) has no FP8 tensor cores. The FP8 weights
# are the right default on Ada/Hopper/Blackwell (L40S, H100, B200); everything
# older needs the bf16 repo. See docfusion.hardware.recommend_model.
OLMOCR_MODEL_FP8 = "allenai/olmOCR-2-7B-1025-FP8"
OLMOCR_MODEL_BF16 = "allenai/olmOCR-2-7B-1025"


class TriageThresholds(BaseModel):
    """Tunable thresholds for the heuristic (model-free) triage pass.

    A page escalates to Tier 2 (VLM) if ANY trigger fires. Defaults are
    calibrated against olmOCR-Bench; see ``docfusion calibrate``.
    """

    min_text_chars: int = Field(120, description="Below this the text layer is unreliable (likely a scan).")
    max_image_area_ratio: float = Field(0.45, description="Image coverage above this suggests a scanned/graphic page.")
    max_math_density: float = Field(0.015, description="Math-symbol chars / total chars above this → math-dense page.")
    max_path_objects: int = Field(220, description="Vector path count above this suggests complex tables/figures.")
    min_avg_word_len: float = Field(2.0, description="Very short 'words' indicate a shredded/garbled text layer.")
    max_chars_per_page: int = Field(
        12000, description="Very dense pages (tiny text) beat the deterministic layout model; escalate."
    )


class VLMEndpoint(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    model: str = OLMOCR_MODEL_FP8
    api_key: str = "docfusion-local"      # vLLM ignores it but the client requires one
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    timeout_s: int = 180

    # Rendering: olmOCR-2 is trained on longest-side-normalised images, not a DPI.
    target_longest_image_dim: int = TARGET_LONGEST_IMAGE_DIM

    # Retry policy: escalating temperature, per upstream. Greedy decoding is what
    # produces repetition loops, so retrying at temperature 0 changes nothing.
    max_attempts: int = Field(
        len(TEMPERATURE_BY_ATTEMPT), ge=1, le=len(TEMPERATURE_BY_ATTEMPT),
        description="Attempts per page along the temperature ladder.",
    )
    use_guided_decoding: bool = Field(
        True, description="Constrain the reply to the YAML front-matter shape via vLLM guided_regex."
    )
    fallback_to_text_layer: bool = Field(
        True, description="On total failure emit the embedded text layer (flagged) instead of raising."
    )

    # Degenerate-generation guards. The span threshold is the load-bearing one:
    # repeat *count* alone flags dot leaders and underscore form fields, and a
    # flagged page is retried the full length of the temperature ladder.
    repetition_ngram_size: int = 64
    repetition_max_repeats: int = 3
    repetition_min_span_chars: int = 200

    # Anchoring is OFF by default: olmOCR-2 is a no-anchoring model and upstream
    # documents its anchor flag as "not used for new models". Enable only when
    # pointing this client at an anchoring-era model (olmOCR v1 / 0725 / 0825).
    use_anchoring: bool = Field(
        False, description="Inject the PDF text layer into the prompt (olmOCR v1-era models only)."
    )
    anchor_max_chars: int = 6000


class PipelineConfig(BaseModel):
    thresholds: TriageThresholds = TriageThresholds()
    vlm: VLMEndpoint = VLMEndpoint()
    use_docling_tier1: bool = Field(
        True, description="Use Docling for Tier-1 extraction when installed; else fall back to the raw text layer."
    )
    docling_ocr: bool = Field(
        False,
        description="Enable Docling's own OCR engine. Off by default: it pulls RapidOCR/PP-OCR "
                    "weights from modelscope.cn at first use, adding an unaudited model family to "
                    "the runtime BOM, and triage already routes weak-text-layer pages to Tier 2.",
    )
    enforce_license_audit: bool = True
    tier2_enabled: bool = Field(
        True, description="When False (e.g. CPU-only dev), all pages go through Tier 1 and "
                          "would-be escalations are only reported, never sent to the VLM.",
    )
    force_tier2_all: bool = Field(
        False,
        description="Send every page to the VLM, bypassing triage. This is the accuracy "
                    "ceiling and the cost ceiling; benchmarking it against the default "
                    "routing is how you measure what triage actually costs you.",
    )
    max_tier2_workers: int = Field(
        4, ge=1, le=64,
        description="Concurrent in-flight page requests. vLLM batches continuously, so a "
                    "serial client leaves most of the GPU idle.",
    )
