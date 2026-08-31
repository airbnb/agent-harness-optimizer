"""Model pricing table and cost estimation for EMNLP efficiency metrics."""

from __future__ import annotations

# (input_price_per_M_tokens, output_price_per_M_tokens) in USD
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic via Bedrock global
    "bedrock/global.anthropic.claude-haiku-4-5-20251001": (1.0, 5.0),
    "bedrock/global.anthropic.claude-sonnet-4-6": (3.0, 15.0),
    "bedrock/global.anthropic.claude-opus-4-7": (5.0, 25.0),
    # Azure OpenAI
    "azure/gpt-5.4-mini": (0.40, 1.60),
    "azure/gpt-4.1-mini": (0.40, 1.60),
    "azure/gpt-4.1": (2.0, 8.0),
    "azure/gpt-4o": (2.5, 10.0),
    "azure/gpt-4o-mini": (0.15, 0.60),
    # Direct Anthropic API
    "anthropic/claude-haiku-4-5-20251001": (1.0, 5.0),
    "anthropic/claude-sonnet-4-6": (3.0, 15.0),
    "anthropic/claude-opus-4-7": (5.0, 25.0),
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost from token counts using MODEL_PRICING table.

    Returns 0.0 for unknown models.
    """
    rate_in, rate_out = MODEL_PRICING.get(model, (0.0, 0.0))
    return (prompt_tokens * rate_in + completion_tokens * rate_out) / 1_000_000
