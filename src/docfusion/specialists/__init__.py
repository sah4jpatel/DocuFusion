"""Per-domain specialist models, routed by region class.

One generalist VLM is not the best answer for every region of a page. olmOCR-2
is excellent at reading order and content fidelity and structurally cannot emit
chart data points, because its prompt asks it to label figures rather than read
them. A chart-derendering model does that one thing far better and costs 300M
parameters instead of 7B.

So the page is segmented first, and each region goes to whatever is best at it:

===============  ==================================  ==========  ===============
region           specialist                          licence     origin
===============  ==================================  ==========  ===============
text             olmOCR-2-7B                         Apache-2.0  Ai2 (US)
table            Table Transformer (TATR) /           MIT         Microsoft (US)
                 Docling TableFormer                 MIT         IBM (US)
chart / figure   DePlot (Pix2Struct)                 Apache-2.0  Google (US)
formula          pix2tex                             MIT         independent
layout           Docling / Granite-Docling-258M      MIT/Apache  IBM (US)
typography       docfusion.formatting (no model)     Apache-2.0  this project
grounding        docfusion.grounding (no model)      Apache-2.0  this project
===============  ==================================  ==========  ===============

Every entry is Apache-2.0 or MIT with no revenue cap and no copyleft. That is
the constraint, and it eliminates otherwise-strong options: Chandra and Surya
are revenue-capped OpenRAIL-M, **texify is GPL-3.0** (copyleft, which is a
different and in some ways worse problem for embedding in a proprietary
pipeline), and MinerU is "Apache-2.0 plus additional terms", which is not
Apache-2.0.

Specialists are optional and lazily loaded. A specialist that is not installed
is simply not used, and the region falls back to the generalist — so the base
install stays light and nothing here can fail closed on a machine that only
wants Markdown.
"""

from docfusion.specialists.base import (
    Region,
    RegionKind,
    Specialist,
    SpecialistResult,
    available_specialists,
    get_specialist,
    register_specialist,
)

__all__ = [
    "Region",
    "RegionKind",
    "Specialist",
    "SpecialistResult",
    "available_specialists",
    "get_specialist",
    "register_specialist",
]
