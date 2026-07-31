"""Serialise all PDFium access behind one process-wide lock.

PDFium keeps global library state and is **not** thread-safe. Opening a
separate ``PdfDocument`` per thread is not sufficient isolation: concurrent
calls corrupt the allocator and the process dies with a native
``munmap_chunk(): invalid pointer`` — no Python traceback, no partial results,
just a dead worker. That is exactly what document-level concurrency produced
here before this lock existed.

Serialising PDFium costs almost nothing in this pipeline because the expensive
part of a Tier-2 page is the *network wait* on the VLM, not the rasterisation:
a page renders in tens of milliseconds and then blocks for seconds on
inference. So the rule is:

    hold the lock for PDFium calls, release it before the HTTP request

which keeps many pages in flight on the GPU while only one thread is ever
inside PDFium.

The lock is reentrant so nested helpers (profile a page inside an already-held
document scope) do not deadlock.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

PDFIUM_LOCK = threading.RLock()


@contextmanager
def pdfium_guard() -> Iterator[None]:
    """Hold the PDFium lock for the duration of the block."""
    with PDFIUM_LOCK:
        yield
