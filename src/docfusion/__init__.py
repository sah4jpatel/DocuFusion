"""DocFusion: license-compliant enterprise document intelligence."""

from docfusion.config import (
    OLMOCR_MODEL_BF16,
    OLMOCR_MODEL_FP8,
    PipelineConfig,
    TriageThresholds,
    VLMEndpoint,
)
from docfusion.hardware import ServingPlan, detect_gpus, plan_serving
from docfusion.licenses import assert_compliant, audit
from docfusion.pipeline import DocFusionPipeline, DocumentResult

__version__ = "0.2.0"
__all__ = [
    "PipelineConfig", "TriageThresholds", "VLMEndpoint",
    "OLMOCR_MODEL_FP8", "OLMOCR_MODEL_BF16",
    "DocFusionPipeline", "DocumentResult",
    "audit", "assert_compliant",
    "detect_gpus", "plan_serving", "ServingPlan",
]
