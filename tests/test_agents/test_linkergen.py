"""Tests for LinkerGenAgent — mocked LangChain, tests SMILES and formula generation."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from agents.agent_2_linkergen.linkergen_agent import LinkerGenAgent, LinkerGenConfig


@pytest.fixture
def mock_llm_response():
    """Mock LangChain LLM response."""
    mock_resp = MagicMock()
    mock_resp.content = "1. CC(=O)O\n2. c1ccccc1\n3. OC(=O)c1ccc(C(=O)O)cc1"
    return mock_resp


@pytest.fixture
def agent_with_mock(mock_llm_response):
    """Create LinkerGenAgent with mocked LLM."""
    with patch(
        "agents.agent_2_linkergen.linkergen_agent.ChatOpenAI"
    ) as MockLLM:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_llm_response
        MockLLM.return_value = mock_instance
        config = LinkerGenConfig(openai_api_key="fake-key")
        agent = LinkerGenAgent(config)
        yield agent


@pytest.fixture
def examples_file(tmp_dir):
    """Create a temporary examples file."""
    f = tmp_dir / "examples.txt"
    f.write_text("OC(=O)c1ccc(C(=O)O)cc1\nOC(=O)c1cccc(C(=O)O)c1")
    return str(f)


class TestLinkerGenConfig:
    def test_defaults(self):
        cfg = LinkerGenConfig()
        assert cfg.model_name == "gpt-4"
        assert cfg.temperature == 1.0


class TestLoadExamples:
    def test_load_existing_file(self, agent_with_mock, examples_file):
        content = agent_with_mock._load_examples(examples_file)
        assert "OC(=O)" in content

    def test_load_missing_file(self, agent_with_mock):
        result = agent_with_mock._load_examples("/nonexistent/path.txt")
        assert result is None


class TestGenerateSmiles:
    def test_returns_string(self, agent_with_mock, examples_file):
        result = agent_with_mock.generate_smiles_from_smiles(
            examples_file, num_linkers=3
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_saves_to_output_file(self, agent_with_mock, examples_file, tmp_dir):
        out = str(tmp_dir / "output.txt")
        agent_with_mock.generate_smiles_from_smiles(
            examples_file, num_linkers=3, output_file=out
        )
        assert Path(out).exists()
        assert len(Path(out).read_text()) > 0

    def test_raises_on_missing_examples(self, agent_with_mock):
        with pytest.raises(ValueError, match="Could not load"):
            agent_with_mock.generate_smiles_from_smiles("/no/file.txt")


class TestGenerateFormula:
    def test_returns_string(self, agent_with_mock, examples_file):
        result = agent_with_mock.generate_formula_from_formula(
            examples_file, num_linkers=5
        )
        assert isinstance(result, str)
