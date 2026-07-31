"""License registry and compliance auditing.

The entire point of docfusion is to get Marker-class quality with only
permissively licensed *model weights*. Code licenses and weight licenses are
tracked separately because they diverge (Marker's code is Apache-2.0 while its
default Surya/Chandra weights are OpenRAIL-M revenue-capped).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class LicenseClass(str, Enum):
    PERMISSIVE = "permissive"          # Apache-2.0 / MIT — unrestricted internal enterprise use
    RESTRICTED = "restricted"          # OpenRAIL-M revenue caps, research-only, etc.
    PROPRIETARY = "proprietary"


@dataclass(frozen=True)
class Component:
    name: str
    kind: str                          # "code" | "weights" | "dataset"
    license: str
    license_class: LicenseClass
    developer: str
    notes: str = ""


REGISTRY: dict[str, Component] = {
    c.name: c
    for c in [
        # ---- code ----
        Component("marker", "code", "Apache-2.0", LicenseClass.PERMISSIVE, "Datalab",
                  "Harness/orchestrator only; safe to vendor and modify."),
        Component("docling", "code", "MIT", LicenseClass.PERMISSIVE, "IBM Research"),
        Component("olmocr-toolkit", "code", "Apache-2.0", LicenseClass.PERMISSIVE, "Ai2"),
        Component("docfusion", "code", "Apache-2.0", LicenseClass.PERMISSIVE, "internal"),
        Component("vllm", "code", "Apache-2.0", LicenseClass.PERMISSIVE, "vLLM project"),
        # ---- weights ----
        Component("chandra", "weights", "AI Pubs OpenRAIL-M (modified)", LicenseClass.RESTRICTED,
                  "Datalab", "Revenue/funding caps; NOT deployable internally without a commercial license."),
        Component("surya", "weights", "OpenRAIL-M (modified)", LicenseClass.RESTRICTED,
                  "Datalab", "Marker's default layout/OCR weights; same revenue caps as Chandra."),
        Component("olmocr-2-7b", "weights", "Apache-2.0", LicenseClass.PERMISSIVE, "Ai2",
                  "e.g. allenai/olmOCR-2-7B-1025-FP8; weights, data and training code all open."),
        Component("docling-layout-heron", "weights", "Apache-2.0 (CDLA-Permissive data)",
                  LicenseClass.PERMISSIVE, "IBM Research", "DocLayNet-family layout model."),
        Component("tableformer", "weights", "MIT", LicenseClass.PERMISSIVE, "IBM Research"),
        # Docling's optional OCR engine. Apache-2.0, but the weights are fetched
        # from modelscope.cn at first use rather than pinned at install time, so
        # it is deliberately excluded from the default BOM and must be cleared
        # explicitly before PipelineConfig.docling_ocr is turned on.
        Component("rapidocr", "weights", "Apache-2.0", LicenseClass.PERMISSIVE, "RapidAI / PaddlePaddle",
                  "PP-OCR weights downloaded from modelscope.cn on first use; permissive but "
                  "unpinned and third-party-hosted. Vendor and pin before enabling docling_ocr."),
    ]
}

# Components docfusion will actually load at runtime, per tier.
DEFAULT_BOM: dict[str, list[str]] = {
    "tier1": ["docfusion", "docling", "docling-layout-heron", "tableformer"],
    "tier2": ["docfusion", "marker", "olmocr-toolkit", "olmocr-2-7b", "vllm"],
}

# Components docfusion explicitly refuses to route to.
DENYLIST: frozenset[str] = frozenset({"chandra", "surya"})


@dataclass
class AuditResult:
    ok: bool
    violations: list[str] = field(default_factory=list)
    bill_of_materials: list[Component] = field(default_factory=list)


def audit(components: list[str] | None = None) -> AuditResult:
    """Verify every component in the (planned) runtime BOM is enterprise-safe."""
    names = components if components is not None else sorted(
        {n for tier in DEFAULT_BOM.values() for n in tier}
    )
    result = AuditResult(ok=True)
    for name in names:
        comp = REGISTRY.get(name)
        if comp is None:
            result.ok = False
            result.violations.append(f"unknown component: {name!r} (not in registry)")
            continue
        result.bill_of_materials.append(comp)
        if name in DENYLIST or comp.license_class is not LicenseClass.PERMISSIVE:
            result.ok = False
            result.violations.append(
                f"{comp.name} [{comp.kind}] is {comp.license} "
                f"({comp.license_class.value}): {comp.notes or 'not cleared for enterprise use'}"
            )
    return result


def assert_compliant(components: list[str] | None = None) -> None:
    res = audit(components)
    if not res.ok:
        raise RuntimeError("License audit failed:\n  - " + "\n  - ".join(res.violations))
