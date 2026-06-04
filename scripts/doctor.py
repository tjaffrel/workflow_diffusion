"""mofgen doctor — diagnose GPU / CUDA / PyTorch install state.

Run with:  pixi run doctor   (or  pixi run -e cuda doctor)
"""

import os
import shutil
import subprocess
import sys


def detect_gpus():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return (
            out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
        )
    except Exception:
        return None


def main():
    print("=== mofgen doctor ===")
    env_name = os.environ.get("PIXI_ENVIRONMENT_NAME", "")
    gpus = detect_gpus()
    if gpus:
        print(f"NVIDIA GPU(s) detected:\n{gpus}")
    else:
        print("No NVIDIA GPU detected (nvidia-smi missing or returned nothing).")

    try:
        import torch
        import torch.version  # explicit submodule import (torch.version.cuda below)
    except Exception as e:  # noqa: BLE001
        print(f"PyTorch import FAILED: {e}")
        print("FIX: run the installer ->  ./install.sh   (or  pixi install)")
        sys.exit(1)

    print(f"PyTorch version:        {torch.__version__}")
    print(f"torch.version.cuda:     {torch.version.cuda}")
    print(f"cuda.is_available():    {torch.cuda.is_available()}")
    print(f"device_count:           {torch.cuda.device_count()}")

    if gpus and torch.version.cuda is None:
        if env_name == "cuda":
            print(
                "\nDIAGNOSIS: GPU present but the cuda env has a CPU build of PyTorch."
            )
            print("FIX: reinstall ->  pixi install -e cuda")
            print("     verify    ->  pixi run -e cuda check-cuda")
            sys.exit(2)
        print(
            "\nNOTE: GPU detected, but this (CPU) environment uses CPU PyTorch by design."
        )
        print(
            "For GPU acceleration:  pixi install -e cuda  &&  pixi run -e cuda doctor"
        )
        return
    if gpus and not torch.cuda.is_available():
        print("\nDIAGNOSIS: GPU + CUDA build present, but CUDA is not available.")
        print("FIX: run inside the cuda env (-e cuda); ensure the NVIDIA driver")
        print("     supports CUDA >= 12 (check `nvidia-smi`).")
        sys.exit(2)
    if not gpus:
        print("\nDIAGNOSIS: CPU-only machine. The default env is correct.")
        print("OK: use  pixi install  and the default tasks.")
        return
    print("\nOK: GPU PyTorch is working.")


if __name__ == "__main__":
    main()
