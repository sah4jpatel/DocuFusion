# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project is pre-1.0, so minor versions may include breaking changes.

## [0.3.0] — 2026-07-31

### Added
- **Model fusion** (`src/docfusion/specialists/`): a region router that segments
  a page from its own PDF object list (no model, ~1ms) and dispatches chart and
  figure regions to per-domain specialists — DePlot (Apache-2.0, Google
  Research) for chart derendering, pix2tex (MIT) for formula recognition.
  Specialists are optional and lazily loaded; fusion only ever appends to the
  generalist's output, never replaces it. Every specialist declares its licence
  at registration and is surfaced by `docfusion audit`.
- Chart/table region discrimination via a lattice test (a ruled table has thin
  lines in both axes; a chart does not) plus text-density filtering, so tables
  are never misrouted to a chart model.
- **Typography recovery** (`src/docfusion/formatting.py`): bold, italic,
  underline, strikethrough and heading level recovered deterministically from
  PDF font metadata and thin-rule detection, then re-applied to olmOCR-2's
  plain-text output. Unicode-tolerant matching (ligatures, quote/dash
  normalisation, soft hyphens) and same-style-only ambiguity resolution.
- **Visual grounding** (`src/docfusion/grounding.py`): block-level bounding
  boxes and reading order reconstructed from per-glyph coordinates, with
  column-gutter detection so multi-column pages are read down rather than
  across.
- **Image input support** (`src/docfusion/io.py`): `.png`/`.jpg`/`.tiff`/etc.
  scans are wrapped into a one-page PDF at the door, so triage, formatting,
  grounding and the CLI all work on them unchanged. Transparent images are
  flattened onto white.
- Lone-surrogate sanitisation: malformed embedded PDF encodings can yield
  unpaired UTF-16 surrogates that crash JSON/UTF-8 serialisation; stripped at
  the point text enters the system.
- Per-page wall-clock budget (`VLMEndpoint.page_budget_s`, default 240s) so one
  pathological page cannot hold a GPU slot indefinitely; the retry ladder now
  logs every invalid attempt instead of failing silently.
- GPU concurrency now sized from measured KV-cache bytes-per-token against a
  *typical* page rather than the context ceiling, and `--gpu-memory-utilization`
  is automatically capped on GPUs that also drive a display (WSL2/WDDM), where
  over-claiming VRAM causes a silent throughput collapse.
- `pyproject.toml` `specialists` extra so chart/formula fusion is actually
  `pip install`-able.
- ParseBench integration (`integrations/parsebench/`) as a versioned provider
  adapter, plus full deployment-readiness verification: wheel build, clean
  install, and CLI smoke test in an isolated venv; full test suite genuinely
  run (not just syntax-checked) on Python 3.10, 3.12, 3.13 and 3.14 across
  Windows and Linux.

### Fixed
- Heading markers were inserted mid-line when a title was split across
  PDF text runs (`Meeting Notice and # Voting # Roadmap`); headings now only
  apply at a line start, spans are merged by line before marking, and
  mid-line "headings" degrade to bold instead.
- Hardware tests were environment-dependent: several called `plan_serving()`
  without pinning `shared_display`, so they silently inherited whatever the
  *actual host* was and failed under WSL while passing on bare metal.

## [0.2.0] — 2026-07-30

### Added
- Corrected the olmOCR-2 client to match the model's actual trained contract:
  the real v4 YAML-front-matter prompt (not an invented one), no anchoring
  (olmOCR-2 is a no-anchoring model), text-before-image ordering, an escalating
  temperature retry ladder, rotation correction, and stripping of the YAML
  front matter the model emits — a client returning `message.content` verbatim
  had been leaking it into every escalated page.
- Process-wide PDFium serialisation (`src/docfusion/pdfium_lock.py`): PDFium
  keeps global state and is not thread-safe; concurrent access previously
  aborted the process with a native `munmap_chunk()` error.
- Document-level batch concurrency (`--doc-workers`), since page-level
  concurrency alone does nothing for a corpus of single-page scans.
- Hardware preflight (`src/docfusion/hardware.py`): detects FP8 tensor-core
  support (Ampere — A100, RTX 3090/A6000 — cannot run the FP8 weights the
  olmOCR README defaults to) and recommends the bf16 build instead.
- Real benchmark harness (`src/docfusion/bench/`) delegating scoring to
  olmOCR's own scorer so results stay comparable to the published leaderboard;
  `scripts/compare_topologies.py` for measuring what triage costs in accuracy
  against an all-VLM ceiling.
- Docling Tier-1 pinned to CPU by default (colocated with vLLM, which
  pre-allocates most VRAM, GPU contention caused Tier-2 timeouts) and its OCR
  engine disabled by default (fetches unaudited weights from `modelscope.cn`
  at conversion time).
- License audit (`docfusion audit`) as a CI-gating job; denylist for
  OpenRAIL-M-restricted weights (Chandra, Surya) and GPL-3.0 code (texify).

### Fixed
- Degenerate-generation guard false-fired on ordinary document runs (table of
  contents dot leaders, form underscores) because it counted repeats without
  checking span length; each false positive burned the full retry ladder.
- The repetition-guard mock server had returned clean Markdown rather than
  olmOCR-2's real front-matter-prefixed reply, so 31 tests passed against a
  client that would have corrupted every escalated page in production.
