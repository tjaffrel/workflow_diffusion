"""Tests for MCP agent tool wrappers — mocked LLM providers."""

import pytest
from unittest.mock import patch, MagicMock


class TestGenerateMof:
    def test_basic_generation(self):
        with patch("mofgen_tools.agents.mof_master.MOFMaster") as MockMaster:
            mock_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.success_count = 1
            mock_response.generation_time = 0.5
            mock_response.mode_used.value = "basic"
            mock_structure = MagicMock()
            mock_structure.formula = "ZnC4H4O4"
            mock_structure.cif_content = "data_test"
            mock_structure.metal_sbu = None
            mock_structure.properties = {}
            mock_response.structures = [mock_structure]
            mock_instance.generate_basic_structures.return_value = mock_response
            MockMaster.return_value = mock_instance

            from mofgen_tools.agents.mof_master import generate_mof
            result = generate_mof("Generate 1 MOF")

            assert result["success_count"] == 1
            assert len(result["structures"]) == 1
            assert result["structures"][0]["formula"] == "ZnC4H4O4"


class TestParseIntent:
    """Tests for _parse_intent metal detection and composition parsing."""

    def test_metal_with_symbol(self):
        from mofgen_tools.agents.mof_master import _parse_intent
        intent = _parse_intent("Generate 3 MOFs with Zr-based SBUs")
        assert intent["mode"] == "metal_specific"
        assert intent["metal"] == "Zr"
        assert intent["count"] == 3

    def test_metal_with_full_name(self):
        from mofgen_tools.agents.mof_master import _parse_intent
        intent = _parse_intent("I want cobalt MOFs")
        assert intent["mode"] == "metal_specific"
        assert intent["metal"] == "Co"

    def test_basic_no_false_positives(self):
        from mofgen_tools.agents.mof_master import _parse_intent
        for text in [
            "Generate diverse MOFs",
            "Give me all MOFs",
            "Generate nice looking MOFs",
            "Generate various MOFs",
        ]:
            intent = _parse_intent(text)
            assert intent["mode"] == "basic", f"False positive for: {text!r}"

    def test_composition_with_ratios(self):
        from mofgen_tools.agents.mof_master import _parse_intent
        intent = _parse_intent(
            "Generate MOFs with composition Zn:0.2, C:0.4, H:0.2, O:0.2"
        )
        assert intent["mode"] == "composition_specific"
        assert intent["composition"]["Zn"] == pytest.approx(0.2)

    def test_composition_without_ratios_falls_to_basic(self):
        from mofgen_tools.agents.mof_master import _parse_intent
        intent = _parse_intent("composition-based MOFs")
        assert intent["mode"] == "basic"

    def test_count_default_is_one(self):
        from mofgen_tools.agents.mof_master import _parse_intent
        intent = _parse_intent("Generate a MOF structure")
        assert intent["count"] == 1

    def test_count_extracted(self):
        from mofgen_tools.agents.mof_master import _parse_intent
        intent = _parse_intent("Generate 5 MOFs")
        assert intent["count"] == 5


class TestAnalyzeStructure:
    def test_file_not_found(self):
        from mofgen_tools.agents.qforge import analyze_structure
        with pytest.raises(FileNotFoundError):
            analyze_structure("/nonexistent/mof.cif")


class TestGenerateLinker:
    def test_calls_agent(self):
        with patch("mofgen_tools.agents.linker_gen.LinkerGenAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.generate_smiles_from_smiles.return_value = "CC(=O)O\nC1=CC=CC=C1"
            MockAgent.return_value = mock_instance

            from mofgen_tools.agents.linker_gen import generate_linker
            result = generate_linker(
                mode="smiles",
                examples_file="examples.csv",
                num_linkers=10,
                provider="openai",
            )

            assert "CC(=O)O" in result
            MockAgent.assert_called_once()
