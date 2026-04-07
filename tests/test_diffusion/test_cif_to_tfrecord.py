"""Tests for CIF to TFRecord conversion — parsing and serialization."""

import pytest
import numpy as np

try:
    from diffuse_materials.cif_to_tfrecord import parse_cif_file, create_tfrecord_example
except ImportError:
    pytest.skip("TensorFlow not available", allow_module_level=True)


class TestParseCif:
    def test_parse_returns_five_elements(self, sample_cif_file):
        result = parse_cif_file(str(sample_cif_file))
        assert len(result) == 5

    def test_frac_coords_shape(self, sample_cif_file):
        frac_coords, _, _, _, _ = parse_cif_file(str(sample_cif_file))
        assert frac_coords.shape == (3, 3)

    def test_atom_types(self, sample_cif_file):
        _, atom_types, _, _, _ = parse_cif_file(str(sample_cif_file))
        assert atom_types.numpy().tolist() == [30, 6, 8]

    def test_lengths(self, sample_cif_file):
        _, _, lengths, _, _ = parse_cif_file(str(sample_cif_file))
        np.testing.assert_allclose(lengths.numpy(), [10.0, 10.0, 10.0])

    def test_angles(self, sample_cif_file):
        _, _, _, angles, _ = parse_cif_file(str(sample_cif_file))
        np.testing.assert_allclose(angles.numpy(), [90.0, 90.0, 90.0])

    def test_formula(self, sample_cif_file):
        _, _, _, _, formula = parse_cif_file(str(sample_cif_file))
        assert isinstance(formula, str)
        assert len(formula) > 0


class TestCreateTfrecordExample:
    def test_creates_valid_example(self, sample_cif_file):
        frac_coords, atom_types, lengths, angles, formula = parse_cif_file(
            str(sample_cif_file)
        )
        example = create_tfrecord_example(
            frac_coords, atom_types, lengths, angles, formula
        )
        serialized = example.SerializeToString()
        assert len(serialized) > 0
