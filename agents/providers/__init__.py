"""LLM provider abstraction for MOFGen agents.

Supports OpenAI and Anthropic. Use :func:`make_provider` to create a
provider by name — it handles per-provider model defaults and lazy imports
so that ``import anthropic`` is not required when using OpenAI only.
"""

from __future__ import annotations

from agents.providers.base import LLMProvider

# Default models per provider.  The pipeline historically used gpt-4o;
# updated to gpt-4.1 (April 2025).
PROVIDER_DEFAULTS: dict[str, str] = {
    "openai": "gpt-4.1",
    "anthropic": "claude-sonnet-4-20250514",
}


def make_provider(
    name: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0,
) -> LLMProvider:
    """Create an LLM provider by name.

    Args:
        name: ``"openai"`` or ``"anthropic"``.
        api_key: API key override (falls back to env var).
        model: Model name override.  *None* uses the provider default.
        temperature: Sampling temperature.
    """
    if name not in PROVIDER_DEFAULTS:
        raise ValueError(
            f"Unknown provider: {name!r}. Use 'openai' or 'anthropic'."
        )
    resolved_model = model or PROVIDER_DEFAULTS[name]

    if name == "openai":
        from agents.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=api_key, model=resolved_model, temperature=temperature,
        )

    from agents.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider(
        api_key=api_key, model=resolved_model, temperature=temperature,
    )


__all__ = ["LLMProvider", "make_provider", "PROVIDER_DEFAULTS"]
