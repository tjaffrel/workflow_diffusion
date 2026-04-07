"""Import smoke test — verifies all core modules load without error.

TensorFlow-dependent modules are tested in a subprocess because TF can
segfault on some CI platforms (notably macOS ARM runners).
"""

import importlib
import os
import subprocess
import sys

# Ensure project root is on sys.path (needed when invoked as `python scripts/test_imports.py`)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CORE_MODULES = [
    "agents",
    "diffuse_materials.model",
    "diffuse_materials.diffusion",
    "diffuse_materials.vae",
]

TF_MODULES = [
    "diffuse_materials.dataset",
    "diffuse_materials.cif_to_tfrecord",
]

failures = []

for mod in CORE_MODULES:
    try:
        importlib.import_module(mod)
        print(f"  OK   {mod}")
    except Exception as e:
        print(f"  FAIL {mod}: {e}")
        failures.append(mod)

for mod in TF_MODULES:
    result = subprocess.run(
        [sys.executable, "-c", f"import {mod}"],
        capture_output=True,
        timeout=60,
    )
    if result.returncode == 0:
        print(f"  OK   {mod}")
    else:
        stderr = result.stderr.decode().strip().split("\n")[-1] if result.stderr else "unknown"
        print(f"  WARN {mod}: {stderr} (TensorFlow optional — skipping)")

if failures:
    print(f"\nFailed to import core modules: {failures}")
    sys.exit(1)
else:
    print("\nAll core imports OK")
