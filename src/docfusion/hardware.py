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

# Measured on olmOCR-2-7B (Qwen2.5-VL-7B, GQA) served bf16 on an RTX 3090:
# 5.25 GB of KV cache held 62,608 tokens, i.e. ~88 KB per token. vLLM reported
# the matching "Maximum concurrency for 16,384 tokens per request: 3.82x".
KV_BYTES_PER_TOKEN = 88_000
# A rendered page is ~1.6k image tokens; a dense enterprise page emits a few
# thousand more. This is what a request actually occupies, as opposed to the
# 16384 ceiling it is merely allowed to reach.
TYPICAL_PAGE_TOKENS = 4096
MIN_NUM_SEQS = 4
MAX_NUM_SEQS = 256

# On a card that also drives a desktop — WSL2/WDDM, or any workstation with a
# monitor attached — the display driver may evict VRAM that another process has
# claimed, paging it to system RAM. There is no error: throughput simply
# collapses. Measured here on an RTX 3090 under WSL2, same model and same
# client, only the utilisation changed:
#     --gpu-memory-utilization 0.87  ->  ~3.5 tok/s per stream, GPU 16% busy
#     --gpu-memory-utilization 0.80  ->  ~47  tok/s per stream, GPU 92% busy
# It is also load-dependent, so it passes in a quiet test and fails later.
SHARED_DISPLAY_MAX_UTILIZATION = 0.80
DEDICATED_UTILIZATION = 0.90


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
    max_num_seqs: int = MAX_NUM_SEQS
    kv_cache_gb: float = 0.0
    shared_display: bool = False
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
            # Sized from the KV cache rather than fixed — see max_num_seqs.
            "--max-num-seqs", str(self.max_num_seqs),
            "--enable-prefix-caching",
        ]
        return args


def concurrency_for(
    kv_cache_gb: float, typical_page_tokens: int = TYPICAL_PAGE_TOKENS
) -> int:
    """How many page requests to allow in flight at once.

    A fixed ``--max-num-seqs 256`` is meaningless on a small card, and a fixed
    small number starves a large one — the first benchmark run here used 8
    while the KV cache sat at 30% utilisation, so throughput was bounded by the
    client rather than the GPU.

    Sized against a *typical* page, not ``max_model_len``. vLLM's own
    "maximum concurrency" line assumes every request occupies the full context,
    which no real page does: a rendered page is ~1.6k image tokens and a dense
    enterprise page emits a few thousand more. Sizing on the worst case gave 4
    on a 3090, while 24 ran that card comfortably. ``--max-num-seqs`` is an
    upper bound, not a reservation — vLLM preempts if a batch genuinely runs
    the cache out — so erring high costs nothing and erring low costs
    throughput.
    """
    if kv_cache_gb <= 0 or typical_page_tokens <= 0:
        return MIN_NUM_SEQS
    tokens = (kv_cache_gb * 1024 ** 3) / KV_BYTES_PER_TOKEN
    return max(MIN_NUM_SEQS, min(MAX_NUM_SEQS, int(tokens // typical_page_tokens)))


def shares_gpu_with_display() -> bool:
    """True when the GPU probably also drives a desktop.

    WSL2 always does — the Windows compositor owns the card and CUDA is a
    guest. Detected from ``/proc/version`` rather than by probing the driver,
    because the failure this guards against is silent and we would rather be
    conservative than fast.
    """
    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


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


def plan_serving(
    gpus: list[GPU] | None = None,
    requested_model: str | None = None,
    shared_display: bool | None = None,
) -> ServingPlan:
    """Produce a serving plan, or explain why this machine cannot serve olmOCR-2."""
    gpus = detect_gpus() if gpus is None else gpus
    plan = ServingPlan(gpus=gpus)

    if shared_display is None:
        shared_display = shares_gpu_with_display()
    plan.shared_display = shared_display
    plan.gpu_memory_utilization = (
        SHARED_DISPLAY_MAX_UTILIZATION if shared_display else DEDICATED_UTILIZATION
    )

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
    plan.kv_cache_gb = round(kv_gb, 2)

    if kv_gb < 3.0:
        plan.max_model_len = 8192
        plan.warnings.append(
            f"Only ~{kv_gb:.1f} GB left for KV cache after weights; capping max-model-len at "
            f"8192 to avoid OOM. Pages needing more context will be flagged, not silently truncated."
        )

    plan.max_num_seqs = concurrency_for(kv_gb)

    if shared_display:
        plan.warnings.append(
            f"This GPU also drives a display (WSL2/WDDM), so utilisation is capped at "
            f"{SHARED_DISPLAY_MAX_UTILIZATION:.2f}. Claiming more lets the display driver page "
            f"the KV cache to system RAM: measured here, 0.87 gave ~3.5 tok/s per stream at 16% "
            f"GPU busy, while 0.80 gave ~47 tok/s at 92%. There is no error when it happens, and "
            f"it only bites once the desktop is busy."
        )

    if len(gpus) > 1 and all(g.name == primary.name for g in gpus):
        plan.warnings.append(
            f"{len(gpus)} identical GPUs detected. olmOCR-2 fits on one card, so prefer "
            f"{len(gpus)} single-GPU replicas over tensor parallelism for batch throughput."
        )

    return plan
