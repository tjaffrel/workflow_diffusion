"""Materials Project data extraction for MOFGen_2025.

Downloads and queries the MOFGen_2025 dataset from MPContribs.

Usage:
    # Full download
    python scripts/mp_data_extraction.py download --output data/mofgen_2025.csv

    # Query subset
    python scripts/mp_data_extraction.py query --metal Zr --output data/zr_mofs.csv
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = "MOFGen_2025"


def get_api_key():
    """Get MP API key from environment."""
    key = os.getenv("MP_API_KEY")
    if not key:
        print(
            "Error: MP_API_KEY not set.\n\n"
            "Set it via environment variable:\n"
            "  export MP_API_KEY='your_key_here'\n\n"
            "Or add it to a .env file in the project root:\n"
            "  MP_API_KEY=your_key_here\n\n"
            "Get your API key at: https://next-gen.materialsproject.org/api#api-key"
        )
        sys.exit(1)
    return key


def get_client(api_key):
    """Create MPContribs client."""
    from mpcontribs.client import Client

    return Client(api_key)


def download(output, format):
    """Download the full MOFGen_2025 dataset."""
    api_key = get_api_key()
    client = get_client(api_key)

    print(f"Fetching contributions from project '{PROJECT_NAME}'...")
    contributions = client.get_contributions(PROJECT_NAME)
    df = pd.json_normalize(contributions)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "json":
        df.to_json(output_path, orient="records", indent=2)
    else:
        csv_path = output_path.with_suffix(".csv")
        json_path = output_path.with_suffix(".json")
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)
        print(f"Saved {len(df)} entries to {csv_path} and {json_path}")
        return df

    print(f"Saved {len(df)} entries to {output_path}")
    return df


def query(metal=None, min_surface_area=None, max_surface_area=None,
          min_pore_size=None, max_pore_size=None, formula=None,
          output="query_results.csv", format="csv"):
    """Query the MOFGen_2025 dataset with filters."""
    api_key = get_api_key()
    client = get_client(api_key)

    print(f"Querying project '{PROJECT_NAME}' with filters...")
    contributions = client.get_contributions(PROJECT_NAME)
    df = pd.json_normalize(contributions)

    if metal:
        metal_col = [c for c in df.columns if "metal" in c.lower()]
        if metal_col:
            df = df[df[metal_col[0]].str.contains(metal, case=False, na=False)]
        else:
            formula_col = [c for c in df.columns if "formula" in c.lower() or "identifier" in c.lower()]
            if formula_col:
                df = df[df[formula_col[0]].str.contains(metal, case=False, na=False)]

    if formula:
        formula_col = [c for c in df.columns if "formula" in c.lower()]
        if formula_col:
            df = df[df[formula_col[0]].str.contains(formula, case=False, na=False)]

    if min_surface_area is not None:
        sa_col = [c for c in df.columns if "surface" in c.lower() and "area" in c.lower()]
        if sa_col:
            df = df[pd.to_numeric(df[sa_col[0]], errors="coerce") >= min_surface_area]

    if max_surface_area is not None:
        sa_col = [c for c in df.columns if "surface" in c.lower() and "area" in c.lower()]
        if sa_col:
            df = df[pd.to_numeric(df[sa_col[0]], errors="coerce") <= max_surface_area]

    if min_pore_size is not None:
        pore_col = [c for c in df.columns if "pore" in c.lower() and ("size" in c.lower() or "diameter" in c.lower())]
        if pore_col:
            df = df[pd.to_numeric(df[pore_col[0]], errors="coerce") >= min_pore_size]

    if max_pore_size is not None:
        pore_col = [c for c in df.columns if "pore" in c.lower() and ("size" in c.lower() or "diameter" in c.lower())]
        if pore_col:
            df = df[pd.to_numeric(df[pore_col[0]], errors="coerce") <= max_pore_size]

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "json":
        df.to_json(output_path, orient="records", indent=2)
    else:
        csv_path = output_path.with_suffix(".csv")
        json_path = output_path.with_suffix(".json")
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)
        print(f"Saved {len(df)} filtered entries to {csv_path} and {json_path}")
        return df

    print(f"Saved {len(df)} filtered entries to {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract MOFGen_2025 data from Materials Project / MPContribs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dl_parser = subparsers.add_parser("download", help="Download full dataset")
    dl_parser.add_argument(
        "--output", default="data/mofgen_2025.csv",
        help="Output file path (default: data/mofgen_2025.csv)"
    )
    dl_parser.add_argument(
        "--format", choices=["csv", "json", "both"], default="both",
        help="Output format (default: both)"
    )

    q_parser = subparsers.add_parser("query", help="Query dataset with filters")
    q_parser.add_argument("--metal", help="Filter by metal element (e.g., Zr, Cu, Zn)")
    q_parser.add_argument("--formula", help="Filter by chemical formula substring")
    q_parser.add_argument("--min-surface-area", type=float, help="Minimum surface area")
    q_parser.add_argument("--max-surface-area", type=float, help="Maximum surface area")
    q_parser.add_argument("--min-pore-size", type=float, help="Minimum pore size/diameter")
    q_parser.add_argument("--max-pore-size", type=float, help="Maximum pore size/diameter")
    q_parser.add_argument(
        "--output", default="data/query_results.csv",
        help="Output file path (default: data/query_results.csv)"
    )
    q_parser.add_argument(
        "--format", choices=["csv", "json", "both"], default="both",
        help="Output format (default: both)"
    )

    args = parser.parse_args()

    if args.command == "download":
        download(output=args.output, format=args.format)
    elif args.command == "query":
        query(
            metal=args.metal,
            min_surface_area=args.min_surface_area,
            max_surface_area=args.max_surface_area,
            min_pore_size=args.min_pore_size,
            max_pore_size=args.max_pore_size,
            formula=args.formula,
            output=args.output,
            format=args.format,
        )


if __name__ == "__main__":
    main()
