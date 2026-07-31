"""Install the DocFusion provider into a ParseBench checkout.

ParseBench discovers providers from a module list and pipelines from a
registration function, both hard-coded in its source. Rather than fork it, this
script patches a checkout in place: it copies the provider module in, adds it
to the discovery list, and registers three pipelines.

    python integrations/parsebench/install.py ~/ParseBench

The patch is idempotent — re-running after a `git pull` in ParseBench reapplies
it cleanly.

Three pipelines are registered, matching the topologies benchmarked on
olmOCR-Bench:

    docfusion_vlm_only    every page through olmOCR-2  (accuracy ceiling)
    docfusion_hybrid      triage routing               (what ships)
    docfusion_tier1_only  Docling only, no GPU         (deterministic floor)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROVIDER_SRC = Path(__file__).parent / "docfusion_provider.py"
PROVIDER_NAME = "docfusion_provider"

PIPELINE_BLOCK = '''
    # --- DocFusion (installed by integrations/parsebench/install.py) ---
    for _df_name, _df_mode in (
        ("docfusion_vlm_only", "vlm_only"),
        ("docfusion_hybrid", "hybrid"),
        ("docfusion_tier1_only", "tier1_only"),
    ):
        register_fn(
            PipelineSpec(
                pipeline_name=_df_name,
                provider_name="docfusion",
                product_type=ProductType.PARSE,
                config={"mode": _df_mode},
            )
        )
'''


def patch_provider_list(parsebench: Path) -> str:
    init = parsebench / "src/parse_bench/inference/providers/parse/__init__.py"
    text = init.read_text(encoding="utf-8")
    if f'"{PROVIDER_NAME}"' in text:
        return "provider list already patched"
    marker = "_PROVIDER_MODULES = ["
    if marker not in text:
        raise SystemExit(f"could not find {marker!r} in {init}")
    text = text.replace(marker, f'{marker}\n    "{PROVIDER_NAME}",', 1)
    init.write_text(text, encoding="utf-8")
    return f"added {PROVIDER_NAME} to provider discovery list"


def patch_pipelines(parsebench: Path) -> str:
    pipelines = parsebench / "src/parse_bench/inference/pipelines/parse.py"
    text = pipelines.read_text(encoding="utf-8")
    if "docfusion_vlm_only" in text:
        return "pipelines already registered"
    marker = "def register_parse_pipelines(register_fn) -> None:  # type: ignore[no-untyped-def]"
    if marker not in text:
        raise SystemExit(f"could not find the pipeline registration function in {pipelines}")
    # Insert immediately after the docstring-free function signature line so the
    # block runs regardless of what else the function does.
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(marker):
            # Skip a docstring if present.
            insert_at = index + 1
            if insert_at < len(lines) and lines[insert_at].lstrip().startswith(('"""', "'''")):
                quote = lines[insert_at].lstrip()[:3]
                insert_at += 1
                while insert_at < len(lines) and quote not in lines[insert_at]:
                    insert_at += 1
                insert_at += 1
            lines.insert(insert_at, PIPELINE_BLOCK)
            break
    else:
        raise SystemExit("registration function not found")
    pipelines.write_text("".join(lines), encoding="utf-8")
    return "registered docfusion_vlm_only / _hybrid / _tier1_only"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    parsebench = Path(sys.argv[1]).expanduser().resolve()
    target = parsebench / "src/parse_bench/inference/providers/parse"
    if not target.is_dir():
        raise SystemExit(f"{parsebench} does not look like a ParseBench checkout")

    shutil.copyfile(PROVIDER_SRC, target / f"{PROVIDER_NAME}.py")
    print(f"copied provider -> {target / f'{PROVIDER_NAME}.py'}")
    print(patch_provider_list(parsebench))
    print(patch_pipelines(parsebench))
    print("\nrun with:")
    print("  parse-bench run docfusion_vlm_only --group tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
