"""Query local extxyz files by property filters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.io import iread


def query_local(
    file: str,
    formula: str | None = None,
    max_energy_per_atom: float | None = None,
    max_force: float | None = None,
    limit: int = 10,
) -> list[dict]:
    """Search a local extxyz file by property filters.

    Stream-reads frames so the whole file is never held in memory.
    """
    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file}")

    results: list[dict] = []
    for atoms in iread(str(path), format="extxyz"):
        f = atoms.get_chemical_formula()
        if formula and formula.lower() not in f.lower():
            continue

        info = atoms.info
        energy = info.get("dft_energy", info.get("energy"))
        n = len(atoms)
        epa = energy / n if energy is not None else None

        if max_energy_per_atom is not None and epa is not None:
            if epa > max_energy_per_atom:
                continue

        forces_arr = atoms.arrays.get("dft_forces", atoms.arrays.get("forces"))
        fmax = float(np.abs(forces_arr).max()) if forces_arr is not None else None

        if max_force is not None and fmax is not None:
            if fmax > max_force:
                continue

        results.append({
            "formula": f,
            "atom_count": n,
            "energy_eV": float(energy) if energy is not None else None,
            "energy_per_atom_eV": float(epa) if epa is not None else None,
            "max_force_eV_per_A": fmax,
        })
        if len(results) >= limit:
            break

    return results
