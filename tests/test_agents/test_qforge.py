"""Tests for MFOModeller — config, structure loading, batch validation."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

try:
    from agents.agent_4_qforge.mof_modeller import MFOModeller, MFOModellerConfig
    HAS_QFORGE = True
except ImportError:
    HAS_QFORGE = False

pytestmark = pytest.mark.skipif(
    not HAS_QFORGE,
    reason="agent_4_qforge dependencies not available (atomate2 API change)",
)


class TestMFOModellerConfig:
    def test_defaults(self):
        cfg = MFOModellerConfig()
        assert cfg.zeopp_path is None
        assert cfg.zeopp_nproc == 3
        assert cfg.sorbates == ["N2", "CO2", "H2O"]
        assert cfg.run_local is True

    def test_custom_config(self):
        cfg = MFOModellerConfig(
            zeopp_path="/usr/local/bin/zeo++",
            zeopp_nproc=8,
            sorbates=["Ar"],
        )
        assert cfg.zeopp_path == "/usr/local/bin/zeo++"
        assert cfg.sorbates == ["Ar"]


class TestMFOModellerInit:
    def test_default_init(self):
        m = MFOModeller()
        assert m.config.run_local is True

    def test_custom_init(self):
        cfg = MFOModellerConfig(store_results=False)
        m = MFOModeller(config=cfg)
        assert m.config.store_results is False


class TestAnalyzeStructure:
    def test_raises_on_invalid_cif(self, tmp_dir):
        bad_cif = tmp_dir / "bad.cif"
        bad_cif.write_text("not a cif file")
        m = MFOModeller()
        with pytest.raises(ValueError, match="Could not parse CIF"):
            m.analyze_structure(bad_cif)


class TestBatchAnalyze:
    def test_raises_on_empty_dir(self, tmp_dir):
        m = MFOModeller()
        with pytest.raises(ValueError, match="No CIF files found"):
            m.batch_analyze(tmp_dir)
