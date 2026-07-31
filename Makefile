.PHONY: setup setup-cpu smoke up down logs test audit preflight triage bench bench-data stress clean

setup:        ; ./setup.sh
setup-cpu:    ; ./setup.sh --cpu
smoke:        ; ./setup.sh --smoke
up:           ; docker compose up -d
down:         ; docker compose down
logs:         ; docker compose logs -f docfusion
audit:        ; docker compose run --rm --entrypoint docfusion docfusion audit
preflight:    ; python3 -m docfusion.cli preflight
test:         ; python3 -m pytest tests/
triage:       ; docker compose run --rm docfusion triage /data/in/$(FILE)
clean:        ; docker compose down -v

# ---- benchmarking ------------------------------------------------------------
# Needs the bench extras and a running vLLM; see README "Measured, not asserted".
BENCH_DIR ?= .bench_data

bench-data:
	hf download allenai/olmOCR-bench --repo-type dataset --local-dir $(BENCH_DIR)

stress:
	python3 scripts/stress_triage.py $(BENCH_DIR) --json triage_stress.json

bench:
	python3 -m docfusion.cli bench $(BENCH_DIR) --json-out benchmark_results.json

# Accuracy vs GPU cost: all-VLM against default routing against Tier-1-only.
compare:
	python3 scripts/compare_topologies.py $(BENCH_DIR) \
	  --categories tables multi_column --workers 8 --out benchmark_results.json
