"""Anthropic LLM provider."""

from __future__ import annotations

import os

from anthropic import Anthropic


class AnthropicProvider:
    """LLM provider backed by the Anthropic API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. "
                "Set ANTHROPIC_API_KEY or pass api_key."
            )
        self.model = model
        self.temperature = temperature
        self.client = Anthropic(api_key=self.api_key)

    def complete(self, system: str, user: str, **kwargs: object) -> str:
        max_tokens = int(kwargs.get("max_tokens", 4000))
        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        return response.content[0].text
