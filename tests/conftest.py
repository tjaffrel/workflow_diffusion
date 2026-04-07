"""Shared test fixtures for MOFGen test suite."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_cif_content():
    """Return a minimal valid CIF string for testing."""
    return """data_test_mof
_cell_length_a   10.0
_cell_length_b   10.0
_cell_length_c   10.0
_cell_angle_alpha   90.0
_cell_angle_beta    90.0
_cell_angle_gamma   90.0
_symmetry_space_group_name_H-M   'P 1'
_symmetry_Int_Tables_number   1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Zn1 Zn 0.0 0.0 0.0
C1 C 0.25 0.25 0.25
O1 O 0.5 0.5 0.5
"""


@pytest.fixture
def sample_cif_file(tmp_dir, sample_cif_content):
    """Write sample CIF content to a temp file and return the path."""
    cif_path = tmp_dir / "test_mof.cif"
    cif_path.write_text(sample_cif_content)
    return cif_path


@pytest.fixture
def mock_openai_response():
    """Return a mock OpenAI chat completion response structure."""
    class MockMessage:
        content = (
            "Structure 1:\n"
            "CIF: data_gen\n_cell_length_a 10.0\n"
            "Formula: ZnC4H4O4\n"
            "Properties: Stable MOF with moderate porosity"
        )

    class MockChoice:
        message = MockMessage()

    class MockCompletion:
        choices = [MockChoice()]

    return MockCompletion()
