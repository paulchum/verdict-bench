from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from benchmarks.t5r.config import SCHEMA_VERSION


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    cached_input_per_mtok: float
    output_per_mtok: float
    source: str


PRICE_SHEET_AS_OF = "2026-07-09"
PRICE_SHEET = {
    "openai/gpt-5-mini-2025-08-07": ModelPrice(
        input_per_mtok=0.25,
        cached_input_per_mtok=0.025,
        output_per_mtok=2.00,
        source="https://developers.openai.com/api/docs/models/gpt-5-mini",
    ),
    "openai/gpt-5-nano-2025-08-07": ModelPrice(
        input_per_mtok=0.05,
        cached_input_per_mtok=0.005,
        output_per_mtok=0.40,
        source="https://developers.openai.com/api/docs/models/gpt-5-nano",
    ),
}

MODEL_ALIASES = {
    "openai/gpt-5-mini": "openai/gpt-5-mini-2025-08-07",
    "openai/gpt-5-nano": "openai/gpt-5-nano-2025-08-07",
}


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def price_sheet_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of": PRICE_SHEET_AS_OF,
        "currency": "USD",
        "unit": "per_million_tokens",
        "models": {model: asdict(price) for model, price in PRICE_SHEET.items()},
    }


def normalize_usage(usage: dict[str, Any] | None, raw_data: dict[str, Any] | None = None) -> Usage:
    """Normalize LiteLLM/OpenAI usage shapes into billed token classes."""
    usage = usage or {}
    raw_usage = {}
    if raw_data:
        raw_usage = raw_data.get("usage") or {}

    prompt_tokens = int(
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or raw_usage.get("prompt_tokens")
        or raw_usage.get("input_tokens")
        or 0
    )
    completion_tokens = int(
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or raw_usage.get("completion_tokens")
        or raw_usage.get("output_tokens")
        or 0
    )

    prompt_details = (
        usage.get("prompt_tokens_details")
        or usage.get("input_tokens_details")
        or raw_usage.get("prompt_tokens_details")
        or raw_usage.get("input_tokens_details")
        or {}
    )
    cached_tokens = int(prompt_details.get("cached_tokens") or 0)
    cached_tokens = min(cached_tokens, prompt_tokens)
    return Usage(
        input_tokens=prompt_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=completion_tokens,
    )


def billed_usd(model: str, usage: Usage) -> float:
    model = MODEL_ALIASES.get(model, model)
    if model not in PRICE_SHEET:
        raise KeyError(f"No pinned price for model {model}")
    price = PRICE_SHEET[model]
    uncached_input = max(usage.input_tokens - usage.cached_input_tokens, 0)
    return (
        uncached_input * price.input_per_mtok
        + usage.cached_input_tokens * price.cached_input_per_mtok
        + usage.output_tokens * price.output_per_mtok
    ) / 1_000_000
