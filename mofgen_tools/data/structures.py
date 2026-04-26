"""CIF extraction from trajectories."""

from __future__ import annotations

from mofgen_tools.data.trajectories import load_trajectory


def get_structure(key: str, frame: str = "last") -> str:
    """Return a CIF string for a specific frame of a trajectory.

    Args:
        key: S3 object key for the trajectory.
        frame: ``"first"``, ``"last"``, or an integer index.

    Returns:
        CIF string.
    """
    traj = load_trajectory(key)
    pmg_traj = traj.to(fmt="pmg")
    n_frames = len(pmg_traj)

    if frame == "first":
        idx = 0
    elif frame == "last":
        idx = -1
    else:
        idx = int(frame)
        if idx < -n_frames or idx >= n_frames:
            raise IndexError(
                f"Frame index {idx} out of range for trajectory "
                f"with {n_frames} frames (use 0..{n_frames - 1})."
            )

    structure = pmg_traj[idx]
    return structure.to(fmt="cif")
