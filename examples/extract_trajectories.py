"""Extract energies, forces, and stresses from MOFGen_2025 relaxation trajectories.

The MOFGen_2025 trajectories are stored as Parquet files on a public S3 bucket
(materialsproject-contribs). This script downloads them and extracts per-frame
energies, forces, and stresses using the ``emmet-core`` RelaxTrajectory model.

See: https://arxiv.org/abs/2504.14110

Usage:
    pixi run extract-trajectories
    pixi run python examples/extract_trajectories.py
    pixi run python examples/extract_trajectories.py --max-trajectories 5
"""

import argparse

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

# VASP uses an internal-stress sign convention (kilobar).
# Multiply by this factor to convert to eV/A^3 with the standard sign.
INTERNAL_KBAR_TO_EV_PER_ANG3 = -1e-22 / elementary_charge


def list_trajectories():
    """Return S3 object summaries for all trajectory Parquet files."""
    s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket(BUCKET_NAME)
    return [
        obj
        for obj in bucket.objects.filter(Prefix=TRAJECTORY_PREFIX)
        if obj.key.endswith(".parquet")
    ]


def load_trajectory(obj) -> RelaxTrajectory:
    """Read a single trajectory from an S3 object."""
    table = pq.read_table(
        f"{obj.bucket_name}/{obj.key}",
        filesystem=s3fs.S3FileSystem(anon=True),
    )
    return RelaxTrajectory.from_arrow(table)


def extract_arrays(traj: RelaxTrajectory):
    """Return energies_per_atom, forces, and stresses arrays from a trajectory."""
    n_atoms = len(traj.elements)
    energies_per_atom = np.array(traj.energy) / n_atoms
    forces = np.array(traj.forces)
    stresses = INTERNAL_KBAR_TO_EV_PER_ANG3 * np.array(traj.stress)
    return energies_per_atom, forces, stresses


def main():
    parser = argparse.ArgumentParser(
        description="Extract forces and stresses from MOFGen_2025 trajectories"
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=1,
        help="Number of trajectories to process (default: 1)",
    )
    args = parser.parse_args()

    print("Listing MOFGen_2025 trajectories on S3...")
    trajectory_objects = list_trajectories()
    print(f"Found {len(trajectory_objects)} trajectories.")

    n = min(args.max_trajectories, len(trajectory_objects))
    for i, obj in enumerate(trajectory_objects[:n]):
        print(f"\n--- Trajectory {i + 1}/{n}: {obj.key} ---")
        traj = load_trajectory(obj)

        energies_per_atom, forces, stresses = extract_arrays(traj)

        print(f"Ionic steps : {len(traj)}")
        print(f"Atoms       : {len(traj.elements)}")
        print(f"Energies    : shape {energies_per_atom.shape}  (eV/atom)")
        print(f"Forces      : shape {forces.shape}  (eV/A)")
        print(f"Stresses    : shape {stresses.shape}  (eV/A^3)")

        print(f"\nFinal energy per atom : {energies_per_atom[-1]:.6f} eV/atom")
        print(f"Max force component   : {np.abs(forces[-1]).max():.6f} eV/A")
        print(f"Stress trace (final)  : {np.trace(stresses[-1]):.6f} eV/A^3")

        # Demonstrate pymatgen / ase conversion
        pmg_traj = traj.to(fmt="pmg")
        print(f"Pymatgen trajectory   : {len(pmg_traj)} frames")
        print(f"  frame keys          : {list(pmg_traj.frame_properties[0].keys())}")

        try:
            ase_traj = traj.to(fmt="ase")
            print(f"ASE trajectory        : {len(ase_traj)} frames")
        except Exception as e:
            print(f"ASE conversion skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
