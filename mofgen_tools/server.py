"""MCP server setup and tool registration for MOFGen Tools."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mofgen")


# ---------------------------------------------------------------------------
# Data tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_trajectories(limit: int | None = None) -> dict:
    """List available MOFGen_2025 DFT relaxation trajectories on S3.

    Returns total count, a sample of trajectory keys, and formulas.
    No credentials required — the data is public.
    """
    from mofgen_tools.data.trajectories import list_trajectories as _list
    return _list(limit=limit)


@mcp.tool()
def search_trajectories(
    formula: str | None = None,
    metal: str | None = None,
    min_atoms: int | None = None,
    max_atoms: int | None = None,
    limit: int = 100,
) -> list[dict]:
    """Filter MOFGen_2025 trajectories by formula, metal element, or atom count range."""
    from mofgen_tools.data.trajectories import search_trajectories as _search
    return _search(
        formula=formula, metal=metal,
        min_atoms=min_atoms, max_atoms=max_atoms,
        limit=limit,
    )


@mcp.tool()
def inspect_trajectory(key: str) -> dict:
    """Get detailed information for one trajectory: ionic steps, elements,
    energies, forces, lattice parameters."""
    from mofgen_tools.data.trajectories import inspect_trajectory as _inspect
    return _inspect(key)


@mcp.tool()
def get_structure(key: str, frame: str = "last") -> str:
    """Return a CIF string for a specific frame of a trajectory.

    ``frame`` can be "first", "last", or an integer index.
    """
    from mofgen_tools.data.structures import get_structure as _get
    return _get(key, frame=frame)


@mcp.tool()
def query_local(
    file: str,
    formula: str | None = None,
    max_energy_per_atom: float | None = None,
    max_force: float | None = None,
    limit: int = 10,
) -> list[dict]:
    """Search a local extxyz file by property filters (formula, energy, force).

    Stream-reads frames so the entire file is never loaded into memory.
    """
    from mofgen_tools.data.local import query_local as _query
    return _query(
        file=file, formula=formula,
        max_energy_per_atom=max_energy_per_atom,
        max_force=max_force, limit=limit,
    )


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

@mcp.tool()
def generate_mof(request: str, provider: str = "openai") -> dict:
    """Invoke MOFMaster to translate a natural language request into
    structured MOF specifications and CIF structures.

    Requires OPENAI_API_KEY or ANTHROPIC_API_KEY depending on provider.
    """
    from mofgen_tools.agents.mof_master import generate_mof as _gen
    return _gen(request=request, provider=provider)


@mcp.tool()
def generate_linker(
    mode: str = "smiles",
    examples_file: str = "",
    num_linkers: int = 50,
    provider: str = "openai",
) -> str:
    """Invoke LinkerGen to generate new MOF linker candidates from examples.

    ``mode`` is "smiles" or "formula". Requires an examples file and API key.
    """
    from mofgen_tools.agents.linker_gen import generate_linker as _gen
    return _gen(
        mode=mode, examples_file=examples_file,
        num_linkers=num_linkers, provider=provider,
    )


@mcp.tool()
def analyze_structure(cif_path: str) -> dict:
    """Invoke QForge to run zeo++/MACE analysis on a MOF CIF file.

    Requires zeo++ to be installed; returns a clear error if unavailable.
    """
    from mofgen_tools.agents.qforge import analyze_structure as _analyze
    return _analyze(cif_path=cif_path)
