"""QForge MCP tool wrapper."""

from __future__ import annotations

from pathlib import Path


def analyze_structure(cif_path: str) -> dict:
    """Invoke QForge to run zeo++/MACE analysis on a MOF structure.

    Args:
        cif_path: Path to CIF file.

    Returns:
        Analysis results dict, or error details if zeo++ is unavailable.
    """
    path = Path(cif_path)
    if not path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    try:
        from agents.agent_4_qforge.mof_modeller import MFOModeller
    except ImportError as e:
        return {
            "error": (
                "QForge dependencies not available. "
                "Requires atomate2, jobflow, and zeo++. "
                f"Import error: {e}"
            )
        }

    modeller = MFOModeller()
    return modeller.analyze_structure(cif_path)
