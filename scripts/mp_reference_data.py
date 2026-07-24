"""Fetch Materials Project reference data used in the MOFGen figures.

Unlike ``mp_data_extraction.py`` (which pulls the contributed *MOFGen_2025*
dataset from MPContribs and therefore needs that project's access), this script
uses only the **standard, public Materials Project API**. Any user with a free
MP API key can reproduce the reference distributions the paper compares against:

  * Formation energy per atom  -> the "Materials Project" reference in the
    formation-energy comparison figure.
  * Bulk modulus (Voigt-Reuss-Hill) -> the elastic-property reference.

No API key is stored in this repository. Provide your own key via the
``MP_API_KEY`` environment variable (or a local ``.env`` file); get one at
https://next-gen.materialsproject.org/api#api-key

Usage:
    export MP_API_KEY="your_key_here"

    # Formation energies (R2SCAN thermo set; optionally restrict elements)
    python scripts/mp_reference_data.py formation-energy \
        --thermo-type R2SCAN --output data/mp_formation_energy.csv

    # Formation energies for MOF-relevant chemistries only
    python scripts/mp_reference_data.py formation-energy \
        --elements Zn,C,H,O --output data/mp_znmof_formation.csv

    # Bulk moduli (elasticity dataset)
    python scripts/mp_reference_data.py bulk-modulus \
        --output data/mp_bulk_modulus.csv
"""

import os
import sys
import argparse
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv is optional
    pass


def get_api_key():
    """Read the MP API key from the environment (never hard-coded)."""
    key = os.getenv("MP_API_KEY")
    if not key:
        print(
            "Error: MP_API_KEY not set.\n\n"
            "Set it via environment variable:\n"
            "  export MP_API_KEY='your_key_here'\n\n"
            "Or add it to a .env file in the project root:\n"
            "  MP_API_KEY=your_key_here\n\n"
            "Get a free key at: https://next-gen.materialsproject.org/api#api-key"
        )
        sys.exit(1)
    return key


def _save(df, output):
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


def formation_energy(thermo_type, elements, output):
    """Pull formation energy per atom from the public MP summary/thermo set."""
    from mp_api.client import MPRester

    kw = dict(fields=["material_id", "formula_pretty", "formation_energy_per_atom"])
    if elements:
        kw["elements"] = elements
    with MPRester(get_api_key()) as mpr:
        # summary carries the ground-state formation energy per material_id
        docs = mpr.materials.summary.search(**kw)
    rows = [
        {
            "material_id": str(d.material_id),
            "formula": d.formula_pretty,
            "formation_energy_eV_per_atom": d.formation_energy_per_atom,
            "reference": f"Materials Project ({thermo_type})",
        }
        for d in docs
        if d.formation_energy_per_atom is not None
    ]
    _save(pd.DataFrame(rows), output)


def bulk_modulus(elements, output):
    """Pull Voigt-Reuss-Hill bulk modulus from the MP elasticity dataset."""
    from mp_api.client import MPRester

    with MPRester(get_api_key()) as mpr:
        if elements:
            # the elasticity endpoint has no element filter; resolve ids via summary
            ids = [str(d.material_id) for d in
                   mpr.materials.summary.search(elements=elements, fields=["material_id"])]
            docs = (mpr.materials.elasticity.search(
                        material_ids=ids, fields=["material_id", "bulk_modulus"])
                    if ids else [])
        else:
            docs = mpr.materials.elasticity.search(fields=["material_id", "bulk_modulus"])
    rows = []
    for d in docs:
        bm = d.bulk_modulus
        vrh = bm.get("vrh") if isinstance(bm, dict) else getattr(bm, "vrh", None)
        if vrh is not None:
            rows.append({
                "material_id": str(d.material_id),
                "bulk_modulus_vrh_GPa": vrh,
                "reference": "Materials Project (elasticity)",
            })
    _save(pd.DataFrame(rows), output)


def _elements(s):
    return [e.strip() for e in s.split(",") if e.strip()] if s else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    fe = sub.add_parser("formation-energy", help="formation energy per atom")
    fe.add_argument("--thermo-type", default="R2SCAN",
                    help="label recorded in the output (e.g. R2SCAN, GGA_GGA+U)")
    fe.add_argument("--elements", type=_elements, default=None,
                    help="comma-separated element filter, e.g. Zn,C,H,O")
    fe.add_argument("--output", default="data/mp_formation_energy.csv")

    bm = sub.add_parser("bulk-modulus", help="Voigt-Reuss-Hill bulk modulus")
    bm.add_argument("--elements", type=_elements, default=None,
                    help="comma-separated element filter")
    bm.add_argument("--output", default="data/mp_bulk_modulus.csv")

    args = p.parse_args()
    if args.command == "formation-energy":
        formation_energy(args.thermo_type, args.elements, args.output)
    elif args.command == "bulk-modulus":
        bulk_modulus(args.elements, args.output)


if __name__ == "__main__":
    main()
