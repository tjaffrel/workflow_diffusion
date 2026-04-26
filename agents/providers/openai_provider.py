"""OpenAI LLM provider."""

from __future__ import annotations

import os

from openai import OpenAI


class OpenAIProvider:
    """LLM provider backed by the OpenAI API.

    The MOFGen pipeline originally used gpt-4o.  Default updated to
    gpt-4.1 (April 2025).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1",
        temperature: float = 0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY or pass api_key."
            )
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key)

    def complete(self, system: str, user: str, **kwargs: object) -> str:
        max_tokens = int(kwargs.get("max_tokens", 4000))
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""
