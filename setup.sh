#!/usr/bin/env bash
# DocFusion one-command setup.
#   ./setup.sh              # GPU stack: vLLM + olmOCR + docfusion (watch mode)
#   ./setup.sh --cpu        # CPU-only dev: Tier 1 pipeline, no VLM
#   ./setup.sh --smoke      # GPU stack + end-to-end smoke conversion
set -euo pipefail
cd "$(dirname "$0")"

MODE="gpu"
SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --cpu) MODE="cpu" ;;
        --smoke) SMOKE=1 ;;
        *) echo "unknown arg: $arg"; exit 2 ;;
    esac
done

say()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- prerequisites -----------------------------------------------------------
command -v docker >/dev/null || fail "docker is not installed (https://docs.docker.com/engine/install/)"
docker compose version >/dev/null 2>&1 || fail "docker compose v2 plugin is required"
docker info >/dev/null 2>&1 || fail "docker daemon not reachable (is it running? are you in the docker group?)"

PICKED_MODEL=""
if [ "$MODE" = "gpu" ]; then
    if ! docker info 2>/dev/null | grep -qi nvidia && ! command -v nvidia-smi >/dev/null; then
        fail "no NVIDIA runtime detected. Install nvidia-container-toolkit, or run ./setup.sh --cpu"
    fi
    if command -v nvidia-smi >/dev/null; then
        vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
        name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        say "GPU: ${name} — ${vram} MiB VRAM, compute capability ${cc}"

        # FP8 tensor cores start at compute capability 8.9 (Ada). A100 is 8.0 and
        # RTX 3090/A6000 are 8.6, so the FP8 build everyone copies from the olmOCR
        # README is the wrong default on some of the most common enterprise cards.
        cc_major=${cc%%.*}; cc_minor=${cc##*.}
        if [ "${cc_major:-0}" -gt 8 ] || { [ "${cc_major:-0}" -eq 8 ] && [ "${cc_minor:-0}" -ge 9 ]; }; then
            PICKED_MODEL="allenai/olmOCR-2-7B-1025-FP8"
            say "FP8 supported — using ${PICKED_MODEL} (~8.5 GB of weights)"
            [ "${vram:-0}" -lt 16000 ] && say "WARN: under 16 GB VRAM; lower VLLM_MAX_MODEL_LEN in .env if vLLM OOMs"
        else
            PICKED_MODEL="allenai/olmOCR-2-7B-1025"
            say "NO FP8 on this card (needs cc >= 8.9) — using bf16 ${PICKED_MODEL} (~16 GB of weights)"
            if [ "${vram:-0}" -lt 20000 ]; then
                fail "bf16 olmOCR-2 needs ~20 GB+; this card has ${vram} MiB. Use an FP8-capable GPU or run ./setup.sh --cpu"
            fi
            [ "${vram:-0}" -lt 24000 ] && say "WARN: tight fit — consider VLLM_MAX_MODEL_LEN=8192 in .env"
        fi
    fi
fi

# ---- environment -------------------------------------------------------------
[ -f .env ] || { cp .env.example .env; say "created .env from .env.example — edit to customize"; }
if [ -n "$PICKED_MODEL" ] && ! grep -q "^OLMOCR_MODEL=${PICKED_MODEL}$" .env 2>/dev/null; then
    # Rewrite rather than append: a stale FP8 line would otherwise win on Ampere.
    if grep -q "^OLMOCR_MODEL=" .env; then
        sed -i.bak "s|^OLMOCR_MODEL=.*|OLMOCR_MODEL=${PICKED_MODEL}|" .env && rm -f .env.bak
    else
        printf 'OLMOCR_MODEL=%s\n' "$PICKED_MODEL" >> .env
    fi
    say "set OLMOCR_MODEL=${PICKED_MODEL} in .env"
fi
mkdir -p data/in data/out
git submodule update --init --depth 1 2>/dev/null || true

# ---- build & launch ----------------------------------------------------------
if [ "$MODE" = "cpu" ]; then
    say "building app image (CPU mode)..."
    docker compose -f docker-compose.yml -f docker-compose.cpu.yml build docfusion
    say "running license audit..."
    docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --entrypoint docfusion docfusion audit
    say "CPU stack ready. Drop PDFs in ./data/in and run:"
    say "  docker compose -f docker-compose.yml -f docker-compose.cpu.yml up docfusion"
    exit 0
fi

say "building app image..."
docker compose build docfusion
say "starting vLLM (first run downloads ~8 GB of olmOCR weights into the hf-cache volume)..."
docker compose up -d vllm
say "waiting for vLLM to become healthy..."
until [ "$(docker inspect -f '{{.State.Health.Status}}' "$(docker compose ps -q vllm)")" = "healthy" ]; do
    sleep 5; printf '.'
done
echo
say "running license audit..."
docker compose run --rm --entrypoint docfusion docfusion audit

if [ "$SMOKE" = 1 ]; then
    say "running end-to-end smoke test..."
    docker compose run --rm --entrypoint python docfusion - <<'PY'
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
c = canvas.Canvas("/data/out/_smoke.pdf", pagesize=letter)
y = 720
for i in range(30):
    c.drawString(72, y, f"f_{i}(x) = ∑ a_i * x^i + ∫ g(t) dt ≈ √(b^2-4ac)/(2a)")
    y -= 20
c.save()
PY
    docker compose run --rm docfusion convert /data/out/_smoke.pdf -o /data/out/_smoke.md \
        --vlm-base-url http://vllm:8000/v1
    say "smoke output:"
    head -5 data/out/_smoke.md
fi

say "starting docfusion in watch mode..."
docker compose up -d docfusion
say "done. Drop PDFs into ./data/in — Markdown appears in ./data/out."
say "  logs:   docker compose logs -f docfusion"
say "  triage: docker compose run --rm docfusion triage /data/in/<file>.pdf"
