#!/usr/bin/env bash
# mofgen bootstrap installer — picks the CPU or GPU pixi environment automatically.
set -euo pipefail

if ! command -v pixi >/dev/null 2>&1; then
  echo "ERROR: pixi is not installed."
  echo "Install it first:  curl -fsSL https://pixi.sh/install.sh | sh"
  echo "Then re-run:        ./install.sh"
  exit 1
fi

os="$(uname -s)"
if [[ "$os" != "Linux" ]]; then
  echo "Non-Linux platform ($os) -> installing the CPU environment (pixi install)"
  pixi install
  echo
  echo "Verify:  pixi run test-imports && pixi run check-tf"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected -> installing the CUDA environment (pixi install -e cuda)"
  pixi install -e cuda
  echo
  echo "Verify GPU:  pixi run -e cuda check-cuda"
else
  echo "No GPU detected -> installing the CPU environment (pixi install)"
  pixi install
  echo
  echo "Verify:  pixi run test-imports && pixi run check-tf"
fi

echo "Diagnose anytime:  pixi run doctor"
