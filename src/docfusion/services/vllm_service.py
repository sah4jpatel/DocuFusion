"""Marker → local vLLM (olmOCR) service shim.

Marker's stock ``OpenAIService`` calls ``client.chat.completions.parse(...,
response_format=<pydantic schema>)`` — OpenAI's proprietary Structured Outputs
contract. Self-hosted vLLM rejects that shape with HTTP 400 ("Input should be
'text' or 'json_object'"). This module provides ``VLLMService``: a drop-in
``BaseService`` subclass that

1. serializes the pydantic ``response_schema`` to JSON Schema and injects it
   into the *prompt* as an instruction,
2. sends a plain ``chat.completions.create`` with ``response_format=
   {"type": "json_object"}`` plus vLLM's ``guided_json`` extra (belt and
   braces — vLLM enforces the schema server-side when supported),
3. robustly extracts/validates the JSON from the reply.

The schema-injection and extraction helpers are plain functions so they can be
unit-tested without importing Marker's heavy dependency tree; the Marker
subclass itself is built lazily via :func:`make_vllm_service`.

Usage with the Marker CLI (after ``pip install docfusion[marker]``)::

    marker_single doc.pdf --use_llm \
        --llm_service docfusion.services.vllm_service.VLLMService \
        --vllm_base_url http://localhost:8000/v1 \
        --vllm_model allenai/olmOCR-2-7B-1025-FP8
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Type

from pydantic import BaseModel

SCHEMA_INSTRUCTION = (
    "Respond ONLY with a single valid JSON object that conforms to this JSON "
    "Schema. No prose, no markdown fences, no explanations.\n\nJSON Schema:\n{schema}"
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def schema_to_json_schema(response_schema: Type[BaseModel]) -> dict[str, Any]:
    return response_schema.model_json_schema()


def inject_schema_into_prompt(prompt: str, response_schema: Type[BaseModel]) -> str:
    schema = json.dumps(schema_to_json_schema(response_schema), indent=2)
    return f"{prompt}\n\n{SCHEMA_INSTRUCTION.format(schema=schema)}"


def extract_json(text: str) -> dict[str, Any]:
    """Tolerant JSON extraction: handles fences, leading prose, trailing junk."""
    if not text:
        raise ValueError("empty model response")
    cleaned = text.strip()
    # strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(cleaned)
    if not match:
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
    return json.loads(match.group(0))


def validate_against_schema(data: dict[str, Any], response_schema: Type[BaseModel]) -> dict[str, Any]:
    """Round-trip through the pydantic model so Marker receives exactly the
    shape ``OpenAIService`` would have returned."""
    return response_schema.model_validate(data).model_dump()


def make_vllm_service():
    """Build the Marker ``BaseService`` subclass (lazy import of marker)."""
    from typing import Annotated, List  # noqa: F401

    import PIL.Image
    from openai import APITimeoutError, OpenAI, RateLimitError
    from marker.schema.blocks import Block
    from marker.services import BaseService

    class VLLMService(BaseService):
        vllm_base_url: Annotated[str, "OpenAI-compatible vLLM endpoint, no trailing slash."] = (
            "http://localhost:8000/v1"
        )
        vllm_model: Annotated[str, "Model name served by vLLM."] = "allenai/olmOCR-2-7B-1025-FP8"
        vllm_api_key: Annotated[str, "Dummy key; vLLM ignores it."] = "docfusion-local"
        vllm_image_format: Annotated[str, "png is safest across VLM chat templates."] = "png"
        vllm_use_guided_json: Annotated[bool, "Also send vLLM guided_json extra."] = True

        def get_client(self) -> OpenAI:
            return OpenAI(base_url=self.vllm_base_url, api_key=self.vllm_api_key)

        def process_images(self, images) -> list[dict]:
            if isinstance(images, PIL.Image.Image):
                images = [images]
            return [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/{};base64,{}".format(
                            self.vllm_image_format,
                            self.img_to_base64(img, format=self.vllm_image_format.upper()),
                        )
                    },
                }
                for img in images
            ]

        def __call__(
            self,
            prompt: str,
            image,
            block: "Block | None",
            response_schema: Type[BaseModel],
            max_retries: int | None = None,
            timeout: int | None = None,
        ):
            max_retries = self.max_retries if max_retries is None else max_retries
            timeout = self.timeout if timeout is None else timeout

            client = self.get_client()
            content = [
                *self.format_image_for_llm(image),
                {"type": "text", "text": inject_schema_into_prompt(prompt, response_schema)},
            ]
            extra_body: dict[str, Any] = {}
            if self.vllm_use_guided_json:
                extra_body["guided_json"] = schema_to_json_schema(response_schema)

            for attempt in range(1, max_retries + 2):
                try:
                    response = client.chat.completions.create(
                        model=self.vllm_model,
                        messages=[{"role": "user", "content": content}],
                        timeout=timeout,
                        temperature=0.0,
                        max_tokens=self.max_output_tokens or 4096,
                        response_format={"type": "json_object"},
                        extra_body=extra_body or None,
                    )
                    text = response.choices[0].message.content
                    data = validate_against_schema(extract_json(text), response_schema)
                    if block is not None and response.usage is not None:
                        block.update_metadata(
                            llm_tokens_used=response.usage.total_tokens, llm_request_count=1
                        )
                    return data
                except (APITimeoutError, RateLimitError, ValueError, json.JSONDecodeError):
                    if attempt >= max_retries + 1:
                        break
                    time.sleep(attempt * self.retry_wait_time)
            return {}

    return VLLMService


def __getattr__(name: str):
    # `--llm_service docfusion.services.vllm_service.VLLMService` resolves here,
    # so Marker is only imported when Marker itself asks for the class.
    if name == "VLLMService":
        return make_vllm_service()
    raise AttributeError(name)
