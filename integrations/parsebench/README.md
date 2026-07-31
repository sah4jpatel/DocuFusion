# ParseBench integration

[ParseBench](https://github.com/run-llama/ParseBench) (LlamaIndex) scores document
parsers on five capability dimensions across ~2,000 human-verified enterprise
pages. Scoring is **fully deterministic** — no LLM judge — so it runs offline
against a local vLLM.

It is a useful complement to olmOCR-Bench because it measures things olmOCR-Bench
does not: chart data extraction, semantic formatting (bold, strikethrough) and
visual grounding (bounding boxes).

## Install

```bash
git clone https://github.com/run-llama/ParseBench ~/ParseBench
cd ~/ParseBench && uv venv && uv pip install -e . && uv pip install -e /path/to/docfusion
python /path/to/docfusion/integrations/parsebench/install.py ~/ParseBench
parse-bench download            # ~2,000 PDFs into ./data
```

`install.py` patches a ParseBench checkout in place — copying the provider in,
adding it to the discovery list and registering three pipelines. It is
idempotent, so re-run it after pulling ParseBench.

## Run

```bash
export DOCFUSION_SERVER_URL=http://localhost:8000/v1
export DOCFUSION_MODEL=allenai/olmOCR-2-7B-1025

parse-bench run docfusion_vlm_only --test --open_report False      # 3 files/category smoke
parse-bench run docfusion_vlm_only --max_concurrent 8 --open_report False
parse-bench run docfusion_hybrid --group table --max_concurrent 8
```

| pipeline | topology |
|---|---|
| `docfusion_vlm_only` | every page through olmOCR-2 — accuracy ceiling |
| `docfusion_hybrid` | triage routing — what ships |
| `docfusion_tier1_only` | Docling only, no GPU — deterministic floor |

ParseBench drives its own concurrency (`--max_concurrent`), so the provider
leaves `max_tier2_workers` at 1 and lets the harness parallelise across pages.

## Two dimensions DocFusion cannot win, and why

These are architectural limits of olmOCR-2, not tuning problems. They are stated
here rather than worked around, because a benchmark you have quietly routed
around is not a measurement.

**Charts.** ParseBench asks for exact data points with series and axis labels —
a test looks like `{"labels": ["IF", "193 UN Member States"], "value": "0.8079"}`.
olmOCR-2's trained prompt tells it to emit a figure *placeholder*
(`![alt](page_startx_starty_width_height.png)`), not chart series. Every
Markdown-linearising model on the leaderboard scores near zero here: Dots.mocr
0.95, DeepSeek-OCR-2 1.1, PaddleOCR-VL 0.9.

**Visual grounding.** This traces every extracted element back to a bounding box.
olmOCR-2 emits linearised Markdown with no coordinates, so there is nothing to
trace. Models without box output score at or near zero (Nemotron 0,
LightOnOCR-2 0, Granite Vision 0).

Closing either gap means adding a component, not tuning DocFusion — a chart-aware
model for the first, a layout detector for the second. Docling's own leaderboard
entry scores 52.76 on charts, so routing chart-heavy pages through Docling's
picture pipeline is the obvious permissively-licensed path if charts matter to
your corpus.
