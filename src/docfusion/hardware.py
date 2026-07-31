"""GPU capability detection and serving-parameter recommendations.

Exists because of one concrete trap: ``allenai/olmOCR-2-7B-1025-FP8`` is the
model everyone copies from the olmOCR README, and it **cannot run natively on
Ampere**. FP8 tensor cores arrived with compute capability 8.9 (Ada), so L40S,
H100 and B200 are fine while A100 (8.0) and RTX 3090/A6000 (8.6) are not — and
A100 is one of the most common enterprise cards. vLLM may fall back to a Marlin
dequantisation kernel, but that path is slower and not what the FP8 build is
for; the bf16 repo is the correct choice below 8.9.

Detection uses ``nvidia-smi`` rather than importing torch so preflight stays
available in the CPU-only install.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field

from docfusion.config import OLMOCR_MODEL_BF16, OLMOCR_MODEL_FP8

FP8_MIN_COMPUTE_CAPABILITY = 8.9

# Weights + activations + CUDA graphs, before any KV cache.
BF16_WEIGHTS_GB = 16.0
FP8_WEIGHTS_GB = 8.5
KV_CACHE_HEADROOM_GB = 4.0


@dataclass
class GPU:
    index: int
    name: str
    memory_total_mb: int
    compute_capability: float

    @property
    def memory_total_gb(self) -> float:
        return self.memory_total_mb / 1024.0

    @property
    def supports_fp8(self) -> bool:
        return self.compute_capability >= FP8_MIN_COMPUTE_CAPABILITY


@dataclass
class ServingPlan:
    gpus: list[GPU] = field(default_factory=list)
    model: str = OLMOCR_MODEL_FP8
    quantization: str = "fp8"
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_vllm_args(self) -> list[str]:
        args = [
            "--model", self.model,
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", f"{self.gpu_memory_utilization:.2f}",
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            # Continuous batching: keep the GPU saturated with page requests.
            "--max-num-seqs", "256",
            "--enable-prefix-caching",
        ]
        return args


def detect_gpus() -> list[GPU]:
    """Enumerate NVIDIA GPUs. Returns [] when none are visible."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []

    gpus: list[GPU] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(GPU(
                index=int(parts[0]),
                name=parts[1],
                memory_total_mb=int(float(parts[2])),
                compute_capability=float(parts[3]),
            ))
        except ValueError:
            continue
    return gpus


def recommend_model(gpu: GPU) -> tuple[str, str]:
    """Pick (model_repo, quantization) appropriate to the card."""
    if gpu.supports_fp8:
        return OLMOCR_MODEL_FP8, "fp8"
    return OLMOCR_MODEL_BF16, "none"


def plan_serving(gpus: list[GPU] | None = None, requested_model: str | None = None) -> ServingPlan:
    """Produce a serving plan, or explain why this machine cannot serve olmOCR-2."""
    gpus = detect_gpus() if gpus is None else gpus
    plan = ServingPlan(gpus=gpus)

    if not gpus:
        plan.errors.append(
            "No NVIDIA GPU detected. Tier 2 needs a GPU; run with --tier1-only for "
            "deterministic CPU extraction."
        )
        return plan

    primary = gpus[0]
    auto_model, auto_quant = recommend_model(primary)
    plan.model, plan.quantization = auto_model, auto_quant
    plan.tensor_parallel_size = 1

    if requested_model and requested_model != auto_model:
        plan.model = requested_model
        if "FP8" in requested_model.upper() and not primary.supports_fp8:
            plan.warnings.append(
                f"{requested_model} is an FP8 build but {primary.name} is compute "
                f"capability {primary.compute_capability} (<{FP8_MIN_COMPUTE_CAPABILITY}). "
                f"FP8 tensor cores are unavailable; vLLM will dequantise through a Marlin "
                f"kernel or fail outright. Prefer {OLMOCR_MODEL_BF16}."
            )
            plan.quantization = "fp8"
        elif "FP8" not in requested_model.upper():
            plan.quantization = "none"

    weights_gb = FP8_WEIGHTS_GB if plan.quantization == "fp8" else BF16_WEIGHTS_GB
    usable_gb = primary.memory_total_gb * plan.gpu_memory_utilization
    kv_gb = usable_gb - weights_gb

    if kv_gb < 1.0:
        needed = weights_gb + KV_CACHE_HEADROOM_GB
        plan.errors.append(
            f"{primary.name} has {primary.memory_total_gb:.0f} GB; {plan.model} needs about "
            f"{needed:.0f} GB (weights {weights_gb:.0f} GB + KV cache). Use the FP8 build on "
            f"a card that supports it, add a second GPU, or run --tier1-only."
        )
        return plan

    # A high-resolution page render is ~1-2k image tokens; the ceiling matters
    # more than throughput here because KV overflow crashes the pod outright.
    if kv_gb < 3.0:
        plan.max_model_len = 8192
        plan.warnings.append(
            f"Only ~{kv_gb:.1f} GB left for KV cache after weights; capping max-model-len at "
            f"8192 to avoid OOM. Pages needing more context will be flagged, not silently truncated."
        )

    if len(gpus) > 1 and all(g.name == primary.name for g in gpus):
        plan.warnings.append(
            f"{len(gpus)} identical GPUs detected. olmOCR-2 fits on one card, so prefer "
            f"{len(gpus)} single-GPU replicas over tensor parallelism for batch throughput."
        )

    return plan
