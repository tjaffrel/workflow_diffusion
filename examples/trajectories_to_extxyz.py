"""Convert MOFGen_2025 DFT trajectories to XYZ formats for MLIP training.

Downloads relaxation trajectories from the public S3 bucket and writes
all ionic steps to XYZ files. Supports three output formats:

  mace    — Extended XYZ with dft_energy/dft_forces/dft_stress (MACE convention)
  extxyz  — Extended XYZ with energy/forces/stress (generic MLIP convention)
  xyz     — Plain XYZ (species + positions only, no forces/energy)

Usage:
    python examples/trajectories_to_extxyz.py --format mace
    python examples/trajectories_to_extxyz.py --format extxyz --output train.xyz
    python examples/trajectories_to_extxyz.py --format xyz --output structures.xyz
    python examples/trajectories_to_extxyz.py --format mace extxyz --workers 8
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

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

FORMATS = ("mace", "extxyz", "xyz")

DEFAULT_OUTPUTS = {
    "mace": "mofgen_2025_mace.xyz",
    "extxyz": "mofgen_2025_extxyz.xyz",
    "xyz": "mofgen_2025.xyz",
}

_s3fs = None


def _get_s3fs():
    global _s3fs
    if _s3fs is None:
        _s3fs = s3fs.S3FileSystem(anon=True)
    return _s3fs


def list_trajectory_keys():
    s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket(BUCKET_NAME)
    return [
        obj.key
        for obj in bucket.objects.filter(Prefix=TRAJECTORY_PREFIX)
        if obj.key.endswith(".parquet")
    ]


def load_trajectory(key: str) -> RelaxTrajectory:
    table = pq.read_table(f"{BUCKET_NAME}/{key}", filesystem=_get_s3fs())
    return RelaxTrajectory.from_arrow(table)


def voigt_from_3x3(stress_3x3):
    """Extract Voigt components (xx, yy, zz, yz, xz, xy) from 3x3 tensor."""
    return [
        stress_3x3[0, 0], stress_3x3[1, 1], stress_3x3[2, 2],
        stress_3x3[1, 2], stress_3x3[0, 2], stress_3x3[0, 1],
    ]


def _formula_from_key(key: str) -> str:
    fname = key.rsplit("/", 1)[-1].replace(".parquet", "")
    return fname.split("--")[0] if "--" in fname else fname


def trajectory_to_mace(traj, pmg_traj, species, energies, forces_all, stress_all, formula):
    """MACE format: dft_energy, dft_forces, dft_stress, config_type, config_weight."""
    n_atoms = len(species)
    buf = StringIO()
    for idx in range(len(traj)):
        struct = pmg_traj[idx]
        lat_str = " ".join(f"{x:.10f}" for x in struct.lattice.matrix.flatten())
        voigt = voigt_from_3x3(stress_all[idx])
        stress_str = " ".join(f"{v:.16e}" for v in voigt)

        buf.write(f"{n_atoms}\n")
        buf.write(
            f'Lattice="{lat_str}" '
            f"Properties=species:S:1:pos:R:3:dft_forces:R:3 "
            f"dft_energy={energies[idx]:.8f} "
            f'dft_stress="{stress_str}" '
            f"formula={formula} "
            f"config_type=mofgen-relax "
            f"config_weight=1.0 "
            f'pbc="T T T"\n'
        )
        for i in range(n_atoms):
            pos = struct.cart_coords[i]
            f = forces_all[idx][i]
            buf.write(
                f"{species[i]:4s} "
                f"{pos[0]:16.8f} {pos[1]:16.8f} {pos[2]:16.8f} "
                f"{f[0]:16.8f} {f[1]:16.8f} {f[2]:16.8f}\n"
            )
    return buf.getvalue()


def trajectory_to_extxyz(traj, pmg_traj, species, energies, forces_all, stress_all, formula):
    """Generic extxyz: energy, forces, stress (compatible with most MLIPs).

    Uses Voigt 6-component stress under the key ``virial_stress`` to avoid
    collision with ASE's reserved ``stress`` key (which expects a 3x3 matrix).
    """
    n_atoms = len(species)
    buf = StringIO()
    for idx in range(len(traj)):
        struct = pmg_traj[idx]
        lat_str = " ".join(f"{x:.10f}" for x in struct.lattice.matrix.flatten())
        voigt = voigt_from_3x3(stress_all[idx])
        stress_str = " ".join(f"{v:.16e}" for v in voigt)

        buf.write(f"{n_atoms}\n")
        buf.write(
            f'Lattice="{lat_str}" '
            f"Properties=species:S:1:pos:R:3:forces:R:3 "
            f"energy={energies[idx]:.8f} "
            f'virial_stress="{stress_str}" '
            f"formula={formula} "
            f'pbc="T T T"\n'
        )
        for i in range(n_atoms):
            pos = struct.cart_coords[i]
            f = forces_all[idx][i]
            buf.write(
                f"{species[i]:4s} "
                f"{pos[0]:16.8f} {pos[1]:16.8f} {pos[2]:16.8f} "
                f"{f[0]:16.8f} {f[1]:16.8f} {f[2]:16.8f}\n"
            )
    return buf.getvalue()


def trajectory_to_xyz(traj, pmg_traj, species, formula, **_kwargs):
    """Plain XYZ: species + Cartesian positions only."""
    n_atoms = len(species)
    buf = StringIO()
    for idx in range(len(traj)):
        struct = pmg_traj[idx]
        buf.write(f"{n_atoms}\n")
        buf.write(f"{formula} frame={idx}\n")
        for i in range(n_atoms):
            pos = struct.cart_coords[i]
            buf.write(
                f"{species[i]:4s} "
                f"{pos[0]:16.8f} {pos[1]:16.8f} {pos[2]:16.8f}\n"
            )
    return buf.getvalue()


FORMAT_WRITERS = {
    "mace": trajectory_to_mace,
    "extxyz": trajectory_to_extxyz,
    "xyz": trajectory_to_xyz,
}


def process_one(key: str, formats: list[str]):
    """Download one trajectory and return format->string dict."""
    traj = load_trajectory(key)
    pmg_traj = traj.to(fmt="pmg")
    species = [str(sp) for sp in pmg_traj[0].species]
    energies = np.array(traj.energy)
    forces_all = np.array(traj.forces)
    stress_all = INTERNAL_KBAR_TO_EV_PER_ANG3 * np.array(traj.stress)
    formula = _formula_from_key(key)

    results = {}
    for fmt in formats:
        writer = FORMAT_WRITERS[fmt]
        results[fmt] = writer(
            traj=traj, pmg_traj=pmg_traj, species=species,
            energies=energies, forces_all=forces_all,
            stress_all=stress_all, formula=formula,
        )
    return results, len(traj)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MOFGen_2025 trajectories to XYZ formats for MLIP training"
    )
    parser.add_argument(
        "--format", nargs="+", choices=FORMATS, default=["mace"],
        help="Output format(s): mace, extxyz, xyz (default: mace). "
             "Multiple formats can be generated in one pass.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file path. When generating multiple formats, this is "
             "used as a base name (e.g. train.xyz -> train_mace.xyz, train_extxyz.xyz). "
             "Defaults per format: " + ", ".join(f"{k}={v}" for k, v in DEFAULT_OUTPUTS.items()),
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel download workers (default: 4)",
    )
    parser.add_argument(
        "--max-trajectories", type=int, default=None,
        help="Limit number of trajectories (default: all)",
    )
    args = parser.parse_args()
    formats = list(dict.fromkeys(args.format))

    # Resolve output paths
    output_paths = {}
    if args.output and len(formats) == 1:
        output_paths[formats[0]] = Path(args.output)
    elif args.output:
        base = Path(args.output)
        stem, suffix = base.stem, base.suffix or ".xyz"
        for fmt in formats:
            output_paths[fmt] = base.parent / f"{stem}_{fmt}{suffix}"
    else:
        for fmt in formats:
            output_paths[fmt] = Path(DEFAULT_OUTPUTS[fmt])

    print("Listing MOFGen_2025 trajectories on S3...")
    keys = list_trajectory_keys()
    print(f"Found {len(keys)} trajectories.")

    if args.max_trajectories:
        keys = keys[: args.max_trajectories]
        print(f"Processing first {len(keys)} trajectories.")

    print(f"Formats: {', '.join(formats)}")
    for fmt, p in output_paths.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {fmt} -> {p}")

    done = 0
    total_frames = 0
    errors = 0
    t0 = time.time()

    file_handles = {fmt: open(p, "w") for fmt, p in output_paths.items()}

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, k, formats): k for k in keys}

            for future in as_completed(futures):
                key = futures[future]
                done += 1
                try:
                    results, n_frames = future.result()
                    for fmt, xyz_str in results.items():
                        file_handles[fmt].write(xyz_str)
                    total_frames += n_frames
                except Exception as e:
                    errors += 1
                    print(f"  ERROR {key}: {e}", file=sys.stderr)

                if done % 50 == 0 or done == len(keys):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(keys) - done) / rate if rate > 0 else 0
                    print(
                        f"  [{done}/{len(keys)}] "
                        f"{total_frames} frames | "
                        f"{errors} errors | "
                        f"{rate:.1f} traj/s | "
                        f"ETA {eta:.0f}s"
                    )
    finally:
        for fh in file_handles.values():
            fh.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s — {done} trajectories, {total_frames} frames, {errors} errors")
    for fmt, p in output_paths.items():
        size_mb = p.stat().st_size / 1e6
        print(f"  {fmt}: {size_mb:.1f} MB -> {p}")


if __name__ == "__main__":
    main()
