# DocFusion application image.
# Contains the triage router, anchoring, vLLM client, CLI, and (optionally)
# the Docling Tier-1 engine. The olmOCR VLM itself runs in the separate
# vllm service (see docker-compose.yml) — this image needs no GPU.
#
#   docker build -t docfusion:latest -f docker/Dockerfile.app .
#   docker build -t docfusion:slim --build-arg WITH_DOCLING=0 -f docker/Dockerfile.app .

FROM python:3.12-slim AS base

ARG WITH_DOCLING=1
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    DOCFUSION_VLM_BASE_URL=http://vllm:8000/v1 \
    DOCFUSION_VLM_MODEL=allenai/olmOCR-2-7B-1025-FP8

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install . \
    && if [ "$WITH_DOCLING" = "1" ]; then pip install ".[docling]"; fi

# Pre-fetch Docling's layout/table models at build time so the container is
# fully offline-capable at runtime (enterprise networks often block HF).
RUN if [ "$WITH_DOCLING" = "1" ]; then \
        python -c "from docling.utils.model_downloader import download_models; download_models()" \
        || echo 'WARN: Docling model prefetch failed (no network at build?); will fetch on first run'; \
    fi

RUN useradd -m -u 10001 docfusion && mkdir -p /data/in /data/out /models \
    && chown -R docfusion /data /models
USER docfusion

COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
CMD ["batch"]
