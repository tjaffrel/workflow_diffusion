"""Tests for LLM provider abstraction."""

import pytest
from unittest.mock import patch, MagicMock

from agents.providers.base import LLMProvider
from agents.providers.openai_provider import OpenAIProvider
from agents.providers.anthropic_provider import AnthropicProvider
from agents.providers import make_provider, PROVIDER_DEFAULTS


class TestLLMProviderProtocol:
    def test_openai_provider_satisfies_protocol(self):
        assert issubclass(OpenAIProvider, LLMProvider)

    def test_anthropic_provider_satisfies_protocol(self):
        assert issubclass(AnthropicProvider, LLMProvider)


class TestMakeProvider:
    def test_creates_openai(self):
        with patch("agents.providers.openai_provider.OpenAI"):
            p = make_provider("openai", api_key="fake")
            assert isinstance(p, OpenAIProvider)
            assert p.model == "gpt-4.1"

    def test_creates_anthropic(self):
        with patch("agents.providers.anthropic_provider.Anthropic"):
            p = make_provider("anthropic", api_key="fake")
            assert isinstance(p, AnthropicProvider)
            assert p.model == "claude-sonnet-4-20250514"

    def test_model_override(self):
        with patch("agents.providers.openai_provider.OpenAI"):
            p = make_provider("openai", api_key="fake", model="gpt-4o")
            assert p.model == "gpt-4o"

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            make_provider("llama")

    def test_defaults_dict_has_both(self):
        assert "openai" in PROVIDER_DEFAULTS
        assert "anthropic" in PROVIDER_DEFAULTS


class TestOpenAIProvider:
    def test_raises_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                OpenAIProvider(api_key="")

    def test_default_model_is_gpt41(self):
        with patch("agents.providers.openai_provider.OpenAI"):
            p = OpenAIProvider(api_key="fake")
            assert p.model == "gpt-4.1"

    def test_complete_calls_openai(self):
        with patch("agents.providers.openai_provider.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_msg = MagicMock()
            mock_msg.content = "response text"
            mock_choice = MagicMock()
            mock_choice.message = mock_msg
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[mock_choice]
            )
            MockOpenAI.return_value = mock_client

            provider = OpenAIProvider(api_key="fake-key", model="gpt-4.1")
            result = provider.complete("system prompt", "user prompt")

            assert result == "response text"
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "gpt-4.1"
            assert call_kwargs["temperature"] == 0
            assert len(call_kwargs["messages"]) == 2

    def test_respects_max_tokens_kwarg(self):
        with patch("agents.providers.openai_provider.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_msg = MagicMock()
            mock_msg.content = "ok"
            mock_choice = MagicMock()
            mock_choice.message = mock_msg
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[mock_choice]
            )
            MockOpenAI.return_value = mock_client

            provider = OpenAIProvider(api_key="fake-key")
            provider.complete("sys", "usr", max_tokens=1000)

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["max_tokens"] == 1000


class TestAnthropicProvider:
    def test_raises_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                AnthropicProvider(api_key="")

    def test_default_model(self):
        with patch("agents.providers.anthropic_provider.Anthropic"):
            p = AnthropicProvider(api_key="fake")
            assert p.model == "claude-sonnet-4-20250514"

    def test_complete_calls_anthropic(self):
        with patch("agents.providers.anthropic_provider.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            mock_block = MagicMock()
            mock_block.text = "claude response"
            mock_client.messages.create.return_value = MagicMock(
                content=[mock_block]
            )
            MockAnthropic.return_value = mock_client

            provider = AnthropicProvider(api_key="fake-key")
            result = provider.complete("system prompt", "user prompt")

            assert result == "claude response"
            mock_client.messages.create.assert_called_once()
            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["system"] == "system prompt"
            assert call_kwargs["temperature"] == 0
            assert call_kwargs["model"] == "claude-sonnet-4-20250514"
