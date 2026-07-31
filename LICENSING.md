# Licensing and bill of materials

DocFusion exists because the highest-scoring open document parsers ship
**permissively licensed code with restrictively licensed weights**. Marker's
harness is Apache-2.0, but its accuracy comes from Chandra and Surya weights
under a modified AI Pubs OpenRAIL-M licence that is free only below a $2–5M
funding/revenue threshold. Above that line you need a commercial agreement, and
"we used the open-source one" stops being true.

Everything DocFusion loads at runtime is Apache-2.0 or MIT, with no revenue cap,
no field-of-use restriction and no reporting obligation.

`docfusion audit` prints this table and **exits non-zero** if anything outside
it enters the runtime BOM. It runs in CI, and the pipeline refuses to construct
if the audit fails.

## Runtime bill of materials

| Component | Kind | Licence | Developer | Country | Notes |
|---|---|---|---|---|---|
| docfusion | code | Apache-2.0 | this project | — | |
| marker | code | Apache-2.0 | Datalab | US | Harness only. Never its weights. |
| docling | code | MIT | IBM Research | US | |
| olmocr-toolkit | code | Apache-2.0 | Ai2 | US | |
| vllm | code | Apache-2.0 | vLLM project | US | |
| olmOCR-2-7B-1025 | **weights** | Apache-2.0 | Ai2 | US | Weights, training code *and* data all open. |
| docling-layout-heron | **weights** | Apache-2.0 | IBM Research | US | DocLayNet-family layout model. |
| tableformer | **weights** | MIT | IBM Research | US | |

Every model in the runtime path is from a US research organisation — Ai2 and
IBM Research — which matters if your supply-chain review asks where weights
came from and who can be subpoenaed about them.

## Explicitly denylisted

These are refused by `docfusion.licenses.DENYLIST`. The audit fails closed if
they appear, so they cannot enter the pipeline by accident.

| Component | Licence | Why it is excluded |
|---|---|---|
| chandra | AI Pubs OpenRAIL-M (modified) | Free only under a $2–5M funding/revenue cap. Above it, commercial licence required. |
| surya | OpenRAIL-M (modified) | Marker's default layout/OCR weights. Same caps. Installing `marker-pdf` will try to fetch these — use Marker only as a `--use_llm` orchestrator. |

## Registered but off by default

| Component | Licence | Why it is not in the default BOM |
|---|---|---|
| rapidocr / PP-OCR | Apache-2.0 | Permissive, but Docling downloads these weights from `modelscope.cn` **at first conversion**, not at install. That is an unaudited host supplying an unpinned model family at runtime — precisely what the audit exists to prevent. Disabled via `PipelineConfig.docling_ocr=False`; triage already routes weak-text-layer pages to Tier 2, so it is also redundant. Vendor and pin the weights before enabling it. |

## Evaluated and rejected

| Candidate | Licence | Verdict |
|---|---|---|
| NVIDIA Nemotron Parse | NVIDIA Nemotron Open Model License | US-developed, small (885M), emits bounding boxes and semantic classes — technically a strong fit for the grounding gap. Rejected as a *default* because it is a vendor-specific licence rather than OSI-approved Apache/MIT, so it needs legal review that Apache-2.0 does not. Reasonable to adopt deliberately; not something to inherit silently. |
| Chandra OCR 2 | OpenRAIL-M | Highest open-weight scorer on several dimensions. Revenue-capped, so out of scope by definition. |
| Non-US open-weight models (MinerU, PaddleOCR-VL, dots.ocr, DeepSeek-OCR, GLM-OCR, Qianfan) | mostly Apache-2.0 | Licences are generally fine. Excluded here only because this project's stated constraint is US-origin weights; if that constraint does not apply to you, several are strong. |

## Datasets

| Dataset | Licence | Use |
|---|---|---|
| olmOCR-mix-1025 | Apache-2.0 (Ai2) | olmOCR-2's training data — disclosed, which is unusual and worth noting for provenance review. |
| olmOCR-Bench | Apache-2.0 (Ai2) | Evaluation only. Not redistributed here; fetched on demand. |
| ParseBench | see dataset card (LlamaIndex) | Evaluation only. Not redistributed here; fetched on demand. |

Benchmark datasets are **not vendored** into this repository. They are
downloaded by `make bench-data` / `parse-bench download` so their licences stay
with their publishers.

## What this does and does not promise

It says: every model weight DocFusion loads by default is under Apache-2.0 or
MIT, from a named US research organisation, with no revenue cap — and CI fails
if that stops being true.

It does not say: this is legal advice, or that your own review is unnecessary.
Licences change, submodules move, and `pip install marker-pdf` will still try to
fetch OpenRAIL-M weights if you let it. Run `docfusion audit` in your own
pipeline and read what it prints.
