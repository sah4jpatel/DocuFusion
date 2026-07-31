"""Render the benchmark results as self-contained SVG charts.

No matplotlib: the output is committed to the repository and rendered by
GitHub's Markdown viewer, so it has to be a plain SVG with no runtime, no
external fonts and no script. Colours are chosen to survive both light and dark
README backgrounds.

    python scripts/make_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
ROOT = Path(__file__).resolve().parents[1]

INK = "#94a3b8"        # axis/label grey, readable on light and dark
OURS = "#0f766e"       # teal — DocFusion
PEER = "#64748b"       # slate — other systems
RESTRICTED = "#b45309" # amber — restricted licence
GRID = "#cbd5e133"


def bar_chart(
    title: str,
    subtitle: str,
    rows: list[tuple[str, float, str]],
    max_value: float = 100.0,
    width: int = 860,
    row_height: int = 30,
    label_width: int = 250,
) -> str:
    """Horizontal bars. rows = [(label, value, colour)]."""
    top = 74
    height = top + len(rows) * row_height + 34
    plot_left = label_width
    plot_width = width - plot_left - 70

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="-apple-system,BlinkMacSystemFont,'
        f'Segoe UI,Helvetica,Arial,sans-serif">',
        f'<text x="0" y="24" font-size="17" font-weight="600" fill="{OURS}">{title}</text>',
        f'<text x="0" y="46" font-size="12.5" fill="{INK}">{subtitle}</text>',
    ]

    # Gridlines every 20 units.
    step = 20
    value = 0
    while value <= max_value:
        x = plot_left + plot_width * value / max_value
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{height - 30}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - 14}" font-size="10.5" fill="{INK}" '
            f'text-anchor="middle">{value}</text>'
        )
        value += step

    for index, (label, val, colour) in enumerate(rows):
        y = top + index * row_height
        bar_w = max(plot_width * (val / max_value), 1.0)
        weight = "600" if colour == OURS else "400"
        parts.append(
            f'<text x="{plot_left - 10}" y="{y + 14}" font-size="12.5" fill="{INK}" '
            f'font-weight="{weight}" text-anchor="end">{label}</text>'
        )
        parts.append(
            f'<rect x="{plot_left}" y="{y + 3}" width="{bar_w:.1f}" height="18" rx="2.5" '
            f'fill="{colour}"/>'
        )
        parts.append(
            f'<text x="{plot_left + bar_w + 7:.1f}" y="{y + 16}" font-size="11.5" '
            f'fill="{INK}" font-weight="{weight}">{val:.1f}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def grouped_chart(
    title: str,
    subtitle: str,
    categories: list[str],
    series: list[tuple[str, list[float], str]],
    width: int = 880,
    max_value: float = 100.0,
) -> str:
    """Vertical grouped bars: categories on x, one bar per series."""
    top, bottom_pad = 92, 58
    plot_h = 250
    height = top + plot_h + bottom_pad
    left = 46
    plot_w = width - left - 20
    group_w = plot_w / len(categories)
    bar_w = min(group_w / (len(series) + 0.7), 40)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="-apple-system,BlinkMacSystemFont,'
        f'Segoe UI,Helvetica,Arial,sans-serif">',
        f'<text x="0" y="24" font-size="17" font-weight="600" fill="{OURS}">{title}</text>',
        f'<text x="0" y="46" font-size="12.5" fill="{INK}">{subtitle}</text>',
    ]

    # Legend
    lx = 0
    for name, _, colour in series:
        parts.append(f'<rect x="{lx}" y="58" width="11" height="11" rx="2" fill="{colour}"/>')
        parts.append(f'<text x="{lx + 16}" y="68" font-size="11.5" fill="{INK}">{name}</text>')
        lx += 20 + len(name) * 6.7

    for value in range(0, int(max_value) + 1, 20):
        y = top + plot_h - plot_h * value / max_value
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - 20}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 3.5:.1f}" font-size="10.5" fill="{INK}" '
            f'text-anchor="end">{value}</text>'
        )

    for c_index, category in enumerate(categories):
        gx = left + c_index * group_w
        for s_index, (_, values, colour) in enumerate(series):
            val = values[c_index]
            bx = gx + (group_w - bar_w * len(series)) / 2 + s_index * bar_w
            bh = plot_h * (val / max_value)
            parts.append(
                f'<rect x="{bx:.1f}" y="{top + plot_h - bh:.1f}" width="{bar_w - 3:.1f}" '
                f'height="{max(bh, 1):.1f}" rx="2" fill="{colour}"/>'
            )
            if val >= 1:
                parts.append(
                    f'<text x="{bx + (bar_w - 3) / 2:.1f}" y="{top + plot_h - bh - 4:.1f}" '
                    f'font-size="9.5" fill="{INK}" text-anchor="middle">{val:.0f}</text>'
                )
        parts.append(
            f'<text x="{gx + group_w / 2:.1f}" y="{top + plot_h + 18}" font-size="11.5" '
            f'fill="{INK}" text-anchor="middle">{category}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    written: list[str] = []

    # --- 1. olmOCR-Bench: what triage costs -------------------------------
    results_path = ROOT / "benchmark_results.json"
    if results_path.exists():
        data = json.loads(results_path.read_text(encoding="utf-8"))
        order = [("vlm_only", "DocFusion — all-VLM"),
                 ("hybrid", "DocFusion — hybrid (default)"),
                 ("tier1_only", "DocFusion — Tier-1 only")]
        rows = [
            (label, data[key]["overall"], OURS if key == "hybrid" else PEER)
            for key, label in order if key in data and data[key].get("overall")
        ]
        rows.append(("Tier-1 only, no Docling", 53.4, PEER))
        svg = bar_chart(
            "olmOCR-Bench — what triage costs",
            "419 tables + multi-column pages, olmOCR-2-7B on one RTX 3090. "
            "Scored by olmOCR's own scorer.",
            rows,
        )
        (DOCS / "chart-olmocr-topologies.svg").write_text(svg, encoding="utf-8")
        written.append("chart-olmocr-topologies.svg")

    # --- 2. Triage escalation by category ---------------------------------
    stress = ROOT / ".bench_data" / "triage_stress.json"
    if stress.exists():
        data = json.loads(stress.read_text(encoding="utf-8"))
        rows = []
        for name, counts in sorted(
            data["by_category"].items(),
            key=lambda kv: -(kv[1]["vlm"] / max(kv[1]["vlm"] + kv[1]["fast"], 1)),
        ):
            pages = counts["vlm"] + counts["fast"]
            rows.append((f"{name}  ({pages}p)", 100 * counts["vlm"] / pages, PEER))
        rows.append((f"ALL  ({data['pages']}p)", 100 * data["overall_escalation"], OURS))
        svg = bar_chart(
            "Triage: how much of each category reaches the GPU",
            "All 1403 olmOCR-Bench pages. 53 pages/sec on CPU, no model. "
            "Escalation rate is the GPU bill.",
            rows,
        )
        (DOCS / "chart-triage-escalation.svg").write_text(svg, encoding="utf-8")
        written.append("chart-triage-escalation.svg")

    # --- 3. ParseBench: per-dimension vs peers ----------------------------
    parse_path = ROOT / "parsebench_results.json"
    if parse_path.exists():
        data = json.loads(parse_path.read_text(encoding="utf-8"))
        cats = ["Tables", "Charts", "Content\nfaithful.", "Semantic\nformatting", "Visual\ngrounding"]
        keys = ["tables", "charts", "content_faithfulness", "semantic_formatting", "visual_grounding"]
        ours = [float(data.get(k) or 0) for k in keys]
        svg = grouped_chart(
            "ParseBench — DocFusion vs open-weight peers",
            "Published leaderboard figures for peers; DocFusion measured here on one RTX 3090.",
            [c.replace("\n", " ") for c in cats],
            [
                ("DocFusion", ours, OURS),
                ("Chandra-2 (revenue-capped)", [89.2, 65.1, 83.7, 61.4, 51.2], RESTRICTED),
                ("Docling-models", [66.4, 52.8, 66.9, 1.0, 66.1], PEER),
            ],
        )
        (DOCS / "chart-parsebench.svg").write_text(svg, encoding="utf-8")
        written.append("chart-parsebench.svg")

    if not written:
        print("no benchmark result files found; nothing to render")
        return 1
    for name in written:
        print(f"wrote docs/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
