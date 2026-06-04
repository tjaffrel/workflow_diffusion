#!/bin/bash
#SBATCH -A m4707
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J mofgen-cpu-test
#SBATCH -o cpu_test_%j.out
#SBATCH -e cpu_test_%j.err
# Smoke runs: submit with `--qos=debug` to override the premium QOS.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p workflow_output

echo "== imports =="; pixi run test-imports
echo "== check-tf =="; pixi run check-tf
echo "== pytest =="; pixi run test
echo "== doctor =="; pixi run doctor || true   # CPU node: expected CPU-only OK path

# Optional online features. Compute nodes may need an outbound proxy.
# Provide one via MOFGEN_HTTPS_PROXY (consult NERSC docs for the current value);
# leave unset to skip the online tests. .env supplies ANTHROPIC_API_KEY.
if [ -n "${MOFGEN_HTTPS_PROXY:-}" ]; then
  export https_proxy="$MOFGEN_HTTPS_PROXY" http_proxy="$MOFGEN_HTTPS_PROXY"
  export HTTPS_PROXY="$MOFGEN_HTTPS_PROXY" HTTP_PROXY="$MOFGEN_HTTPS_PROXY"
fi
if [ -f .env ]; then set -a; . ./.env; set +a; fi

if curl -sI --max-time 15 https://api.anthropic.com >/dev/null 2>&1; then
  echo "== S3 trajectory extraction =="; pixi run extract-trajectories || echo "S3 extraction FAILED — review log"
  if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "== agent (anthropic) =="; pixi run python example_mof_generation.py --provider anthropic || echo "agent run FAILED — review log"
  else
    echo "SKIP agent test: ANTHROPIC_API_KEY not set (.env)."
  fi
else
  echo "SKIP online tests: no outbound HTTPS (set MOFGEN_HTTPS_PROXY to enable)."
fi
echo "CPU feature test complete."
