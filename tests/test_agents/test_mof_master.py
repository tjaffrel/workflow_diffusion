"""Tests for MOFMaster agent — mocked OpenAI, tests data flow and parsing."""

import pytest
from unittest.mock import patch, MagicMock

from agents.mof_master import (
    MOFMaster,
    MOFGenerationMode,
    MOFGenerationRequest,
    MOFGenerationResponse,
    MOFStructure,
)


@pytest.fixture
def mocked_master(mock_openai_response):
    """Create a MOFMaster with mocked OpenAI client."""
    with patch("agents.mof_master.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        MockOpenAI.return_value = mock_client
        master = MOFMaster(api_key="fake-key")
        yield master


class TestMOFStructureDataclass:
    def test_create_basic(self):
        s = MOFStructure(cif_content="data_test", formula="ZnO")
        assert s.cif_content == "data_test"
        assert s.formula == "ZnO"
        assert s.metal_sbu is None

    def test_create_with_metadata(self):
        s = MOFStructure(
            cif_content="data_test",
            formula="ZnO",
            metal_sbu="Zn",
            properties={"pore_size": 5.0},
            generation_metadata={"mode": "basic"},
        )
        assert s.metal_sbu == "Zn"
        assert s.properties["pore_size"] == 5.0


class TestMOFGenerationRequest:
    def test_basic_request(self):
        req = MOFGenerationRequest(mode=MOFGenerationMode.BASIC, count=3)
        assert req.count == 3
        assert req.metal is None

    def test_metal_request(self):
        req = MOFGenerationRequest(
            mode=MOFGenerationMode.METAL_SPECIFIC, count=1, metal="Zr"
        )
        assert req.metal == "Zr"


class TestMOFMasterInit:
    def test_raises_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                MOFMaster()


class TestMOFMasterGeneration:
    def test_generate_basic(self, mocked_master):
        response = mocked_master.generate_basic_structures(count=1)
        assert isinstance(response, MOFGenerationResponse)
        assert response.success_count == 1
        assert response.mode_used == MOFGenerationMode.BASIC
        assert len(response.structures) == 1

    def test_generate_metal_specific(self, mocked_master):
        response = mocked_master.generate_metal_specific_structures(
            metal="Zn", count=2
        )
        assert response.success_count == 2
        assert response.mode_used == MOFGenerationMode.METAL_SPECIFIC
        for s in response.structures:
            assert s.metal_sbu == "Zn"

    def test_generate_composition_specific(self, mocked_master):
        comp = {"Zn": 0.3, "C": 0.4, "O": 0.3}
        response = mocked_master.generate_composition_specific_structures(
            composition=comp, count=1
        )
        assert response.success_count == 1
        assert response.structures[0].composition == comp

    def test_generation_time_recorded(self, mocked_master):
        response = mocked_master.generate_basic_structures(count=1)
        assert response.generation_time >= 0

    def test_metadata_contains_model(self, mocked_master):
        response = mocked_master.generate_basic_structures(count=1)
        assert "model_used" in response.metadata


class TestParseStructureResponse:
    def test_parses_cif_from_response(self, mocked_master):
        text = (
            "Structure 1:\n"
            "CIF: data_gen\n"
            "_cell_length_a 10.0\n"
            "Formula: ZnC4H4O4\n"
            "Properties: Stable MOF"
        )
        s = mocked_master._parse_structure_response(text, 1, "basic")
        assert "data_gen" in s.cif_content
        assert s.formula == "ZnC4H4O4"

    def test_placeholder_on_empty_response(self, mocked_master):
        s = mocked_master._parse_structure_response("No CIF here", 1, "basic")
        assert "data_1" in s.cif_content
        assert s.formula == "ZnC4H4O4"
