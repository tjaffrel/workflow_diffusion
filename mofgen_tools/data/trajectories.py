"""S3 trajectory browsing: list, search, inspect."""

from __future__ import annotations

import re
from functools import lru_cache

import boto3
import numpy as np
import pyarrow.parquet as pq
import s3fs
from botocore import UNSIGNED
from botocore.client import Config
from emmet.core.trajectory import RelaxTrajectory
from scipy.constants import elementary_charge

BUCKET_NAME = "materialsproject-contribs"
TRAJECTORY_PREFIX = "MOFGen_2025/trajectories/"
INTERNAL_KBAR_TO_EV_PER_ANG3 = -1e-22 / elementary_charge

_s3fs: s3fs.S3FileSystem | None = None


def _get_s3fs() -> s3fs.S3FileSystem:
    global _s3fs
    if _s3fs is None:
        _s3fs = s3fs.S3FileSystem(anon=True)
    return _s3fs


def _formula_from_key(key: str) -> str:
    fname = key.rsplit("/", 1)[-1].replace(".parquet", "")
    return fname.split("--")[0] if "--" in fname else fname


def _elements_from_formula(formula: str) -> list[str]:
    return re.findall(r"[A-Z][a-z]?", formula)


@lru_cache(maxsize=1)
def _list_keys() -> list[str]:
    """List all trajectory parquet keys on S3 (cached per session)."""
    s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket(BUCKET_NAME)
    return [
        obj.key
        for obj in bucket.objects.filter(Prefix=TRAJECTORY_PREFIX)
        if obj.key.endswith(".parquet")
    ]


@lru_cache(maxsize=32)
def load_trajectory(key: str) -> RelaxTrajectory:
    """Load a trajectory from S3.  Cached (up to 32) to avoid re-downloading
    the same parquet when a user calls inspect then get_structure."""
    table = pq.read_table(f"{BUCKET_NAME}/{key}", filesystem=_get_s3fs())
    return RelaxTrajectory.from_arrow(table)


def list_trajectories(limit: int | None = None) -> dict:
    """List available trajectories with summary statistics."""
    keys = _list_keys()
    sample = keys[:limit] if limit is not None else keys[:20]
    formulas = [_formula_from_key(k) for k in sample]
    return {
        "total_count": len(keys),
        "sample_keys": sample,
        "sample_formulas": formulas,
    }


def search_trajectories(
    formula: str | None = None,
    metal: str | None = None,
    min_atoms: int | None = None,
    max_atoms: int | None = None,
    limit: int = 100,
) -> list[dict]:
    """Filter trajectories by formula substring, metal element, or atom count."""
    keys = _list_keys()
    results = []
    for key in keys:
        f = _formula_from_key(key)
        if formula and formula.lower() not in f.lower():
            continue
        if metal:
            elems = _elements_from_formula(f)
            if metal not in elems:
                continue
        atom_count = sum(
            int(n) if n else 1
            for n in re.findall(r"[A-Z][a-z]?(\d*)", f)
        )
        if min_atoms is not None and atom_count < min_atoms:
            continue
        if max_atoms is not None and atom_count > max_atoms:
            continue
        results.append({
            "key": key,
            "formula": f,
            "estimated_atom_count": atom_count,
        })
        if len(results) >= limit:
            break
    return results


def inspect_trajectory(key: str) -> dict:
    """Get detailed information for one trajectory."""
    traj = load_trajectory(key)
    pmg_traj = traj.to(fmt="pmg")
    species = [str(sp) for sp in pmg_traj[0].species]
    elements = sorted(set(species))
    energies = np.array(traj.energy)
    forces = np.array(traj.forces)
    stresses = INTERNAL_KBAR_TO_EV_PER_ANG3 * np.array(traj.stress)
    lattice = pmg_traj[-1].lattice

    return {
        "formula": _formula_from_key(key),
        "ionic_steps": len(traj),
        "atom_count": len(species),
        "elements": elements,
        "energy_first_eV": float(energies[0]),
        "energy_last_eV": float(energies[-1]),
        "max_force_final_eV_per_A": float(np.abs(forces[-1]).max()),
        "stress_trace_final_eV_per_A3": float(np.trace(stresses[-1])),
        "lattice_a": float(lattice.a),
        "lattice_b": float(lattice.b),
        "lattice_c": float(lattice.c),
        "lattice_alpha": float(lattice.alpha),
        "lattice_beta": float(lattice.beta),
        "lattice_gamma": float(lattice.gamma),
    }
