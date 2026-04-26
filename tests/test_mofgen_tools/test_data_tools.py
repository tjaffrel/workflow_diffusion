"""Tests for MCP data tools — mocked S3/boto3."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the lru_cache between tests."""
    from mofgen_tools.data import trajectories
    trajectories._list_keys.cache_clear()
    yield
    trajectories._list_keys.cache_clear()


FAKE_KEYS = [
    "MOFGen_2025/trajectories/Zn4C8H4O8--mp-1234.parquet",
    "MOFGen_2025/trajectories/Cu2C6O4--mp-5678.parquet",
    "MOFGen_2025/trajectories/Zr6O8C24--mp-9012.parquet",
]


def _mock_list_keys():
    return FAKE_KEYS


class TestListTrajectories:
    @patch("mofgen_tools.data.trajectories._list_keys", _mock_list_keys)
    def test_returns_total_count(self):
        from mofgen_tools.data.trajectories import list_trajectories
        result = list_trajectories()
        assert result["total_count"] == 3

    @patch("mofgen_tools.data.trajectories._list_keys", _mock_list_keys)
    def test_limit_samples(self):
        from mofgen_tools.data.trajectories import list_trajectories
        result = list_trajectories(limit=1)
        assert len(result["sample_keys"]) == 1


class TestSearchTrajectories:
    @patch("mofgen_tools.data.trajectories._list_keys", _mock_list_keys)
    def test_filter_by_metal(self):
        from mofgen_tools.data.trajectories import search_trajectories
        results = search_trajectories(metal="Zn")
        assert len(results) == 1
        assert "Zn" in results[0]["formula"]

    @patch("mofgen_tools.data.trajectories._list_keys", _mock_list_keys)
    def test_filter_by_formula(self):
        from mofgen_tools.data.trajectories import search_trajectories
        results = search_trajectories(formula="Cu")
        assert len(results) == 1

    @patch("mofgen_tools.data.trajectories._list_keys", _mock_list_keys)
    def test_no_match(self):
        from mofgen_tools.data.trajectories import search_trajectories
        results = search_trajectories(metal="Ag")
        assert len(results) == 0


class TestFormulaFromKey:
    def test_with_double_dash(self):
        from mofgen_tools.data.trajectories import _formula_from_key
        assert _formula_from_key("prefix/Zn4C8--mp-1.parquet") == "Zn4C8"

    def test_without_double_dash(self):
        from mofgen_tools.data.trajectories import _formula_from_key
        assert _formula_from_key("prefix/Zn4C8.parquet") == "Zn4C8"


class TestQueryLocal:
    def test_file_not_found(self):
        from mofgen_tools.data.local import query_local
        with pytest.raises(FileNotFoundError):
            query_local("/nonexistent/file.xyz")
