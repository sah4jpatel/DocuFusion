# Contributing

## Setup

```bash
git clone --recurse-submodules https://github.com/sah4jpatel/DocuFusion
cd DocuFusion
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

The core package is deliberately light — `pydantic`, `pypdfium2`, `pillow`,
`openai`, `pyyaml`. Triage, typography recovery, grounding and the vLLM client
all run with no GPU and no model downloads, so the full test suite passes on a
laptop in under a minute.

## The rules that matter here

**Fixtures must imitate the dependency, not your expectations of it.** The
suite once passed 31 tests against a client that leaked YAML front matter into
every page, because the mock returned the clean Markdown the code expected
rather than what vLLM actually returns. `tests/conftest.py` now reproduces
olmOCR-2's real reply format and vLLM's real HTTP 400. Keep it that way.

**Pin upstream assumptions with tests, not comments.**
`test_prompt_matches_upstream` compares our vendored olmOCR-2 prompt against
`third_party/olmocr` and fails if upstream changes it;
`test_marker_still_uses_structured_outputs` fails if Marker stops needing our
shim. When an assumption about someone else's code is load-bearing, make CI
tell you when it stops being true.

**Measure claims about accuracy.** "This should be better" is not a reason to
merge. `make compare` scores topologies against olmOCR-Bench using upstream's
own scorer; `integrations/parsebench` scores against ParseBench. If a change is
supposed to improve extraction, show the number.

**Never enter a restricted weight into the BOM.** `docfusion audit` runs in CI
and fails closed. If you add a model, register it in `src/docfusion/licenses.py`
with its real licence and update [LICENSING.md](LICENSING.md).

## Things that will bite you

- **PDFium is not thread-safe.** It keeps global state; a per-thread
  `PdfDocument` is not sufficient isolation and concurrent access aborts the
  process with a native `munmap_chunk(): invalid pointer` and no traceback. Take
  `docfusion.pdfium_lock.pdfium_guard()` inside any function that touches
  PDFium — inside, not at the call site, because one unguarded caller corrupts
  the heap for every thread.
- **Write files as UTF-8 explicitly.** Extracted documents are full of
  em-dashes, Greek letters and LaTeX; Windows' locale codec raises on all of
  them. CI runs on Windows for this reason.
- **Don't hold the PDFium lock across a network call.** Rasterising is
  milliseconds, inference is seconds; holding the lock over the request
  serialises the whole pipeline.

## Style

`ruff check src/ tests/ scripts/ integrations/ --select E,F,W,I,B --line-length 120`

Comments should explain *why*, especially where the code looks odd — most odd
code here is odd because of a specific measured failure, and the comment is the
only record of it.
