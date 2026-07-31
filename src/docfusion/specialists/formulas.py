"""Mathematical expression recognition: region image → LaTeX.

olmOCR-2 is genuinely good at inline maths in running text — 83.0 on
olmOCR-Bench's ArXiv Math split — so this specialist is not here to replace it.
It is here for the cases a page-level linearising model handles worst: a
display equation inside a scanned page, a formula in a table cell, or a page so
dense the generalist truncates before reaching it.

``pix2tex`` is MIT and small. ``UniMERNet`` (Apache-2.0, OpenDataLab) scores
higher on hard real-world formulas and is registered as an alternative — but it
is a Chinese-origin model, so it is opt-in rather than the default, to keep the
default BOM US/EU-origin. Both are permissive; the distinction here is
provenance, not licence.

Deliberately *not* used: ``texify``, which is the obvious candidate by
capability but is **GPL-3.0**. Copyleft is a different problem from the
OpenRAIL revenue cap and, for a library meant to be embedded in proprietary
enterprise pipelines, a worse one.
"""

from __future__ import annotations

import logging
import os

from docfusion.specialists.base import (
    Region,
    RegionKind,
    SpecialistResult,
    register_specialist,
)

logger = logging.getLogger(__name__)

UNIMERNET_MODEL = os.getenv("DOCFUSION_FORMULA_MODEL", "wanderkid/unimernet_base")


def wrap_latex(latex: str, display: bool = True) -> str:
    """Wrap bare LaTeX in the delimiters olmOCR-2 uses elsewhere.

    The rest of the pipeline emits ``\\( \\)`` and ``\\[ \\]`` because that is
    what olmOCR-2's prompt specifies, so a formula specialist that emitted
    ``$...$`` would produce a document with two conventions in it.
    """
    latex = (latex or "").strip()
    if not latex:
        return ""
    if latex.startswith(("\\(", "\\[", "$")):
        return latex
    return f"\\[{latex}\\]" if display else f"\\({latex}\\)"


class Pix2TexSpecialist:
    """Formula image → LaTeX (MIT, lukas-blecher/LaTeX-OCR)."""

    name = "pix2tex"
    kinds = (RegionKind.FORMULA,)
    licence = "MIT"
    origin = "independent (EU)"

    def __init__(self) -> None:
        self._model = None

    def available(self) -> bool:
        try:
            import pix2tex  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self) -> None:
        if self._model is None:
            from pix2tex.cli import LatexOCR

            self._model = LatexOCR()

    def run(self, region: Region) -> SpecialistResult:
        if region.image is None:
            return SpecialistResult(specialist=self.name, note="no image supplied")
        try:
            self._load()
            latex = self._model(region.image)
        except Exception as exc:  # noqa: BLE001 — one bad formula must not fail the page
            logger.warning("pix2tex failed on page %d region %d: %s",
                           region.page_index, region.reading_order, exc)
            return SpecialistResult(specialist=self.name, degraded=True, note=str(exc))

        display = region.h > 0.02          # a tall region is a display equation
        return SpecialistResult(markdown=wrap_latex(latex, display=display), specialist=self.name)


class UniMERNetSpecialist:
    """Formula image → LaTeX (Apache-2.0, OpenDataLab). Opt-in: non-US origin."""

    name = "unimernet"
    kinds = (RegionKind.FORMULA,)
    licence = "Apache-2.0"
    origin = "OpenDataLab (CN)"

    def __init__(self, model_name: str = UNIMERNET_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._device = None

    def available(self) -> bool:
        # Opt-in: never used unless explicitly enabled, because its provenance
        # is outside the default US/EU constraint even though its licence is fine.
        if os.getenv("DOCFUSION_ENABLE_UNIMERNET", "0") != "1":
            return False
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True
        ).to(device).eval()
        self._device = device

    def run(self, region: Region) -> SpecialistResult:
        if region.image is None:
            return SpecialistResult(specialist=self.name, note="no image supplied")
        try:
            self._load()
            import torch

            inputs = self._processor(images=region.image, return_tensors="pt").to(self._device)
            with torch.inference_mode():
                generated = self._model.generate(**inputs, max_new_tokens=512)
            latex = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("unimernet failed: %s", exc)
            return SpecialistResult(specialist=self.name, degraded=True, note=str(exc))
        return SpecialistResult(
            markdown=wrap_latex(latex, display=region.h > 0.02), specialist=self.name
        )


@register_specialist("pix2tex", kinds=(RegionKind.FORMULA,), licence="MIT",
                     origin="independent (EU)")
def _make_pix2tex() -> Pix2TexSpecialist:
    return Pix2TexSpecialist()


@register_specialist("unimernet", kinds=(RegionKind.FORMULA,), licence="Apache-2.0",
                     origin="OpenDataLab (CN)")
def _make_unimernet() -> UniMERNetSpecialist:
    return UniMERNetSpecialist()
