"""Serving-plan tests.

The trap these exist for: ``allenai/olmOCR-2-7B-1025-FP8`` is what the olmOCR
README shows, and it needs compute capability 8.9. A100 is 8.0 and RTX
3090/A6000 are 8.6, so the copy-paste default silently misbehaves on some of
the most common enterprise cards.
"""

from __future__ import annotations

from docfusion.config import OLMOCR_MODEL_BF16, OLMOCR_MODEL_FP8
from docfusion.hardware import GPU, concurrency_for, plan_serving, recommend_model

L40S = GPU(index=0, name="NVIDIA L40S", memory_total_mb=46068, compute_capability=8.9)
A100 = GPU(index=0, name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920, compute_capability=8.0)
RTX3090 = GPU(index=0, name="NVIDIA GeForce RTX 3090", memory_total_mb=24576, compute_capability=8.6)
T4 = GPU(index=0, name="Tesla T4", memory_total_mb=15360, compute_capability=7.5)
H100 = GPU(index=0, name="NVIDIA H100 80GB HBM3", memory_total_mb=81559, compute_capability=9.0)


class TestModelSelection:
    def test_ada_and_newer_get_fp8(self):
        assert recommend_model(L40S) == (OLMOCR_MODEL_FP8, "fp8")
        assert recommend_model(H100) == (OLMOCR_MODEL_FP8, "fp8")

    def test_ampere_gets_bf16(self):
        """A100 and 3090 have no FP8 tensor cores."""
        assert recommend_model(A100) == (OLMOCR_MODEL_BF16, "none")
        assert recommend_model(RTX3090) == (OLMOCR_MODEL_BF16, "none")

    def test_requesting_fp8_on_ampere_warns(self):
        plan = plan_serving([A100], requested_model=OLMOCR_MODEL_FP8)
        assert plan.model == OLMOCR_MODEL_FP8
        assert any("FP8" in w and "8.9" in w for w in plan.warnings), plan.warnings

    def test_requesting_fp8_on_ada_is_silent(self):
        plan = plan_serving([L40S], requested_model=OLMOCR_MODEL_FP8)
        assert plan.warnings == []
        assert plan.ok


class TestMemoryPlanning:
    def test_small_card_is_rejected_with_guidance(self):
        """16 GB cannot hold 16 GB of bf16 weights plus a KV cache."""
        plan = plan_serving([T4])
        assert not plan.ok
        assert any("tier1-only" in e for e in plan.errors)

    def test_24gb_ampere_holds_full_context(self):
        """3090/A5000: 16 GB of bf16 weights still leave ~5 GB of KV cache.

        Measured on a real RTX 3090 at --gpu-memory-utilization 0.92: weights
        take 15.63 GiB and vLLM reports a 72,944-token KV cache, i.e. 4.45
        concurrent requests at the full 16384 context. Enough to keep the GPU
        busy, so capping the context here would cost throughput for nothing.
        """
        plan = plan_serving([RTX3090])
        assert plan.ok
        assert plan.model == OLMOCR_MODEL_BF16
        assert plan.max_model_len == 16384

    def test_20gb_ampere_caps_context(self):
        """RTX A4500: bf16 weights leave ~2 GB, so cap rather than OOM mid-batch."""
        a4500 = GPU(index=0, name="NVIDIA RTX A4500", memory_total_mb=20480, compute_capability=8.6)
        plan = plan_serving([a4500])
        assert plan.ok
        assert plan.max_model_len == 8192
        assert any("max-model-len" in w for w in plan.warnings)

    def test_large_card_keeps_full_context(self):
        plan = plan_serving([L40S])
        assert plan.ok and plan.max_model_len == 16384

    def test_no_gpu_is_an_error_not_a_crash(self):
        plan = plan_serving([])
        assert not plan.ok
        assert any("No NVIDIA GPU" in e for e in plan.errors)

    def test_multi_gpu_prefers_replicas_over_tensor_parallel(self):
        plan = plan_serving([L40S, L40S])
        assert plan.tensor_parallel_size == 1
        assert any("replicas" in w for w in plan.warnings)


class TestServeArgs:
    def test_args_carry_throughput_settings(self):
        args = plan_serving([L40S]).as_vllm_args()
        assert "--enable-prefix-caching" in args
        assert args[args.index("--max-model-len") + 1] == "16384"
        assert int(args[args.index("--max-num-seqs") + 1]) > 8

    def test_capped_context_reaches_the_args(self):
        a4500 = GPU(index=0, name="NVIDIA RTX A4500", memory_total_mb=20480, compute_capability=8.6)
        args = plan_serving([a4500]).as_vllm_args()
        assert args[args.index("--max-model-len") + 1] == "8192"


class TestConcurrencySizing:
    """--max-num-seqs is bounded by KV cache, not by a nice round number.

    Fixed at 256 it is meaningless on a small card; fixed at 8 it starved a
    real benchmark run whose KV cache sat at 30% utilisation.
    """

    def test_sized_for_a_typical_page_not_the_context_ceiling(self):
        """Measured: a 3090 ran 24 concurrent streams comfortably.

        vLLM's own "maximum concurrency: 3.82x" assumes every request fills the
        16384 context, which no real page does. Sizing on that worst case
        returned 4 and would have throttled a card that was fine at 24.
        """
        assert concurrency_for(5.25) >= 12
        # Sizing on the full context is what produced the too-low answer.
        assert concurrency_for(5.25, typical_page_tokens=16384) < 8

    def test_denser_pages_allow_fewer_streams(self):
        assert concurrency_for(5.25, 8192) < concurrency_for(5.25, 2048)

    def test_large_cache_is_capped_not_unbounded(self):
        assert concurrency_for(400.0) == 256

    def test_never_returns_zero(self):
        assert concurrency_for(0.01) == 4
        assert concurrency_for(0.0) == 4

    def test_plan_reports_kv_cache_size(self):
        plan = plan_serving([RTX3090])
        assert plan.kv_cache_gb > 0
        assert plan.max_num_seqs >= 4


class TestSharedDisplay:
    """A GPU that also drives a desktop must not be filled to the brim.

    WDDM will page a CUDA process's memory out to system RAM when the compositor
    wants VRAM, with no error and no log line — throughput just collapses.
    Measured on this machine: 0.87 utilisation gave ~3.5 tok/s per stream at 16%
    GPU busy; 0.80 gave ~47 tok/s at 92%. It is load-dependent, so it passes on a
    quiet desktop and fails later under real use.
    """

    def test_shared_display_caps_utilisation(self):
        plan = plan_serving([RTX3090], shared_display=True)
        assert plan.gpu_memory_utilization == 0.80
        assert plan.shared_display

    def test_dedicated_gpu_uses_full_budget(self):
        plan = plan_serving([L40S], shared_display=False)
        assert plan.gpu_memory_utilization == 0.90
        assert not plan.shared_display

    def test_shared_display_is_explained_not_silent(self):
        plan = plan_serving([RTX3090], shared_display=True)
        assert any("display" in w and "tok/s" in w for w in plan.warnings), plan.warnings

    def test_cap_reaches_the_serve_args(self):
        args = plan_serving([RTX3090], shared_display=True).as_vllm_args()
        assert args[args.index("--gpu-memory-utilization") + 1] == "0.80"
