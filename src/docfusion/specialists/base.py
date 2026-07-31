"""Specialist protocol and registry.

A specialist takes one *region* of a page — a crop plus its metadata — and
returns Markdown or HTML for that region. It is deliberately a narrow contract:
specialists must be swappable, because the best model for tables in 2026 will
not be the best model for tables in 2027, and the whole point of this design is
that replacing one does not disturb the others.

Registration is by :class:`RegionKind`, and every specialist declares its
licence. :func:`available_specialists` filters to the ones actually installed,
so an environment that never installed the chart model simply routes charts to
the generalist instead of failing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class RegionKind(str, Enum):
    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    FIGURE = "figure"
    FORMULA = "formula"
    UNKNOWN = "unknown"


@dataclass
class Region:
    """A rectangular area of a page handed to a specialist.

    Coordinates are normalised to ``[0, 1]`` with a top-left origin, matching
    :mod:`docfusion.grounding`. ``image`` is a PIL image of the crop, rendered
    by the caller so specialists never touch PDFium (which is serialised).
    """

    kind: RegionKind
    page_index: int
    x: float
    y: float
    w: float
    h: float
    image: object | None = None      # PIL.Image.Image, kept untyped to avoid the import
    text_layer: str = ""             # what the PDF itself says here, when anything
    reading_order: int = 0

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)


@dataclass
class SpecialistResult:
    """What a specialist produced for one region."""

    markdown: str = ""
    confidence: float | None = None
    specialist: str = ""
    degraded: bool = False
    note: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.markdown.strip())


@runtime_checkable
class Specialist(Protocol):
    """The whole contract. Anything satisfying this can be registered."""

    name: str
    kinds: tuple[RegionKind, ...]
    licence: str
    origin: str

    def available(self) -> bool:
        """True when this specialist's dependencies and weights are present."""
        ...

    def run(self, region: Region) -> SpecialistResult:
        ...


@dataclass
class _Entry:
    factory: Callable[[], Specialist]
    kinds: tuple[RegionKind, ...]
    licence: str
    origin: str
    instance: Specialist | None = None


_REGISTRY: dict[str, _Entry] = {}
# Cache availability: probing usually means importing torch, which is slow
# enough that doing it per region would dominate a page.
_AVAILABILITY: dict[str, bool] = {}


def register_specialist(
    name: str,
    kinds: tuple[RegionKind, ...],
    licence: str,
    origin: str,
) -> Callable[[Callable[[], Specialist]], Callable[[], Specialist]]:
    """Register a specialist factory under ``name``.

    The licence is required, not optional metadata. It is surfaced by
    ``docfusion audit`` alongside the rest of the BOM, so a specialist cannot
    enter the pipeline without its licence being visible.
    """
    def decorator(factory: Callable[[], Specialist]) -> Callable[[], Specialist]:
        if name in _REGISTRY:
            raise ValueError(f"specialist {name!r} is already registered")
        _REGISTRY[name] = _Entry(factory=factory, kinds=kinds, licence=licence, origin=origin)
        return factory
    return decorator


def get_specialist(name: str) -> Specialist | None:
    """Instantiate a registered specialist, or None if it is unavailable."""
    entry = _REGISTRY.get(name)
    if entry is None:
        return None
    if entry.instance is None:
        try:
            entry.instance = entry.factory()
        except Exception as exc:  # noqa: BLE001 — a missing optional dep is not an error
            logger.info("specialist %s unavailable: %s", name, exc)
            _AVAILABILITY[name] = False
            return None
    if _AVAILABILITY.get(name) is None:
        try:
            _AVAILABILITY[name] = entry.instance.available()
        except Exception:  # noqa: BLE001
            _AVAILABILITY[name] = False
    return entry.instance if _AVAILABILITY.get(name) else None


def available_specialists(kind: RegionKind | None = None) -> dict[str, Specialist]:
    """Every installed specialist, optionally filtered to one region kind."""
    found: dict[str, Specialist] = {}
    for name, entry in _REGISTRY.items():
        if kind is not None and kind not in entry.kinds:
            continue
        instance = get_specialist(name)
        if instance is not None:
            found[name] = instance
    return found


def registry_bom() -> list[dict[str, object]]:
    """Licence bill of materials for every registered specialist."""
    rows: list[dict[str, object]] = []
    for name, entry in sorted(_REGISTRY.items()):
        rows.append({
            "name": name,
            "kinds": [k.value for k in entry.kinds],
            "licence": entry.licence,
            "origin": entry.origin,
            "installed": get_specialist(name) is not None,
        })
    return rows


@dataclass
class FusionReport:
    """What the router did with a page, for cost and quality accounting."""

    regions: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    by_specialist: dict[str, int] = field(default_factory=dict)
    fell_back: int = 0
    notes: list[str] = field(default_factory=list)
