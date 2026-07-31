"""Input normalisation: accept what enterprises actually have.

Two defects this fixes, both found by running a real benchmark rather than by
reading the code.

**Images are documents too.** The whole pipeline opens its input with PDFium,
so a scanned page delivered as ``.png``, ``.jpg`` or ``.tiff`` failed with
``PDFium: Data format error`` and the document was recorded as a failure. 42 of
ParseBench's 2,079 inputs are images, and a fax/scanner archive is mostly
images. Rather than teach triage, formatting, grounding and rendering each to
handle bitmaps, an image is wrapped into a one-page PDF once, at the door —
after which every downstream stage works unchanged.

**Lone surrogates crash serialisation.** PDFium returns raw UTF-16 code units,
and a malformed embedded encoding yields unpaired surrogates (U+D800–U+DFFF).
Python holds them happily in a ``str`` and then raises
``UnicodeEncodeError: surrogates not allowed`` the moment anything writes JSON
or UTF-8 — which is to say, at the API boundary, after all the GPU work is
done. They are stripped where the text enters the system.
"""

from __future__ import annotations

import logging
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Formats PIL can open and we can wrap into a PDF.
IMAGE_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif", ".ppm",
})

# Unpaired UTF-16 surrogates. Valid inside a Python str, fatal on encode.
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def sanitize_text(text: str) -> str:
    """Drop lone surrogates so the result can actually be encoded.

    Deliberately silent and lossy: the alternative is an exception at the
    serialisation boundary, and a page that is missing one undecodable glyph is
    far more useful than a batch job that died writing its output.
    """
    if not text:
        return text
    if not _SURROGATE_RE.search(text):
        return text
    cleaned = _SURROGATE_RE.sub("", text)
    logger.debug("stripped %d lone surrogate(s)", len(text) - len(cleaned))
    return cleaned


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def image_to_pdf_bytes(path: str | Path) -> bytes:
    """Wrap a bitmap into a single-page PDF at its natural size."""
    import io as _io

    from PIL import Image

    with Image.open(path) as image:
        # PDF has no alpha channel; flatten onto white so transparent scans do
        # not come out as black rectangles.
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        buffer = _io.BytesIO()
        image.save(buffer, format="PDF", resolution=72.0)
        return buffer.getvalue()


@contextmanager
def as_pdf(path: str | Path) -> Iterator[Path]:
    """Yield a PDF path for ``path``, converting images on the way in.

    A converted file lives in a temporary directory for the duration of the
    context and is removed afterwards, so callers never have to know whether
    the original input was a PDF.
    """
    source = Path(path)
    if not is_image(source):
        yield source
        return

    with tempfile.TemporaryDirectory(prefix="docfusion-") as directory:
        target = Path(directory) / f"{source.stem}.pdf"
        target.write_bytes(image_to_pdf_bytes(source))
        logger.info("converted %s to a single-page PDF for processing", source.name)
        yield target
