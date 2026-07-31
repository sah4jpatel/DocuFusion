#!/usr/bin/env bash
# DocFusion container entrypoint.
#   batch            convert /data/in/**.pdf -> /data/out/**.md (default)
#   watch            batch on a loop (poll every $DOCFUSION_WATCH_INTERVAL seconds)
#   triage <pdf>     routing report only (no VLM needed)
#   audit            license audit
#   <anything else>  passed through to `docfusion` verbatim
set -euo pipefail

VLM_URL="${DOCFUSION_VLM_BASE_URL:-http://vllm:8000/v1}"
VLM_MODEL="${DOCFUSION_VLM_MODEL:-allenai/olmOCR-2-7B-1025-FP8}"
WAIT_SECS="${DOCFUSION_VLM_WAIT:-600}"
EXTRA_ARGS=()
[ "${DOCFUSION_NO_DOCLING:-0}" = "1" ] && EXTRA_ARGS+=(--no-docling)
[ "${DOCFUSION_TIER1_ONLY:-0}" = "1" ] && EXTRA_ARGS+=(--tier1-only)
[ -n "${DOCFUSION_WORKERS:-}" ] && EXTRA_ARGS+=(--workers "${DOCFUSION_WORKERS}")

wait_for_vlm() {
    echo "[docfusion] waiting for vLLM at ${VLM_URL} (up to ${WAIT_SECS}s; model load can be slow on first run)..."
    local start=$SECONDS
    until curl -fsS "${VLM_URL}/models" >/dev/null 2>&1; do
        if (( SECONDS - start > WAIT_SECS )); then
            echo "[docfusion] ERROR: vLLM not reachable after ${WAIT_SECS}s" >&2
            exit 1
        fi
        sleep 5
    done
    echo "[docfusion] vLLM is up: $(curl -fsS "${VLM_URL}/models" | head -c 300)"
}

cmd="${1:-batch}"
case "$cmd" in
    batch)
        docfusion audit
        wait_for_vlm
        exec docfusion batch /data/in /data/out \
            --vlm-base-url "$VLM_URL" --vlm-model "$VLM_MODEL" "${EXTRA_ARGS[@]}"
        ;;
    watch)
        docfusion audit
        wait_for_vlm
        interval="${DOCFUSION_WATCH_INTERVAL:-30}"
        echo "[docfusion] watching /data/in every ${interval}s"
        while true; do
            docfusion batch /data/in /data/out --skip-existing \
                --vlm-base-url "$VLM_URL" --vlm-model "$VLM_MODEL" "${EXTRA_ARGS[@]}" || true
            sleep "$interval"
        done
        ;;
    triage|audit)
        shift || true
        exec docfusion "$cmd" "$@"
        ;;
    *)
        exec docfusion "$@"
        ;;
esac
