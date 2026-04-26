"""LLM provider protocol for framework-agnostic agent execution."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers used by MOFGen agents."""

    def complete(self, system: str, user: str, **kwargs: object) -> str:
        """Send a system + user message and return the assistant text."""
        ...
