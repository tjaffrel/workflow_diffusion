"""Tests for MP data extraction — mocked MPContribs client."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from scripts.mp_data_extraction import get_api_key, download, query


class TestGetApiKey:
    def test_returns_key_from_env(self):
        with patch.dict("os.environ", {"MP_API_KEY": "test-key-123"}):
            assert get_api_key() == "test-key-123"

    def test_exits_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(SystemExit):
                get_api_key()


@pytest.fixture
def mock_contributions():
    """Sample MPContribs response data."""
    return [
        {
            "identifier": "mp-12345",
            "formula": "Zn4O(BDC)3",
            "data.metal": "Zn",
            "data.surface_area": 1200.5,
            "data.pore_diameter": 8.3,
        },
        {
            "identifier": "mp-67890",
            "formula": "Zr6O4(BDC)6",
            "data.metal": "Zr",
            "data.surface_area": 2500.0,
            "data.pore_diameter": 12.1,
        },
        {
            "identifier": "mp-11111",
            "formula": "Cu3(BTC)2",
            "data.metal": "Cu",
            "data.surface_area": 800.0,
            "data.pore_diameter": 5.5,
        },
    ]


@pytest.fixture
def mock_client(mock_contributions):
    """Mock MPContribs client."""
    with patch("scripts.mp_data_extraction.get_client") as mock_get:
        client = MagicMock()
        client.get_contributions.return_value = mock_contributions
        mock_get.return_value = client
        yield client


class TestDownload:
    def test_saves_csv(self, mock_client, tmp_dir):
        with patch.dict("os.environ", {"MP_API_KEY": "fake"}):
            out = str(tmp_dir / "out.csv")
            df = download(output=out, format="csv")
            assert Path(out).exists()
            assert len(df) == 3

    def test_saves_json(self, mock_client, tmp_dir):
        with patch.dict("os.environ", {"MP_API_KEY": "fake"}):
            out = str(tmp_dir / "out.json")
            df = download(output=out, format="json")
            assert Path(out).exists()

    def test_saves_both(self, mock_client, tmp_dir):
        with patch.dict("os.environ", {"MP_API_KEY": "fake"}):
            out = str(tmp_dir / "out.csv")
            df = download(output=out, format="both")
            assert (tmp_dir / "out.csv").exists()
            assert (tmp_dir / "out.json").exists()


class TestQuery:
    def test_filter_by_formula_metal(self, mock_client, tmp_dir):
        with patch.dict("os.environ", {"MP_API_KEY": "fake"}):
            out = str(tmp_dir / "q.csv")
            df = query(formula="Zr", output=out, format="csv")
            assert len(df) >= 1
