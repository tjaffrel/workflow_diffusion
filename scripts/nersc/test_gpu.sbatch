#!/bin/bash
#SBATCH -A m4707_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 00:15:00
#SBATCH -J mofgen-gpu-test
#SBATCH -o gpu_test_%j.out
#SBATCH -e gpu_test_%j.err
# On Perlmutter, submit -q premium with -C gpu (it maps to the gpu_premium QOS);
# premium is whole-node (4 GPUs). Smoke runs: submit with `--qos=debug`.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p workflow_output

echo "== nvidia-smi =="; nvidia-smi
echo "== doctor =="; pixi run -e cuda doctor
echo "== check-cuda =="; pixi run -e cuda check-cuda
echo "== assert CUDA available =="
pixi run -e cuda python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK on', torch.cuda.get_device_name(0))"
echo "== pytest (cuda env) =="; pixi run -e cuda test
echo "GPU feature test complete."
