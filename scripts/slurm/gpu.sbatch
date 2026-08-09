#!/bin/bash
#SBATCH --job-name=cotmon
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.log
#
# Generic single-H100 launcher for the cotmon pipeline on GT PACE ICE.
# Config gleaned from the user's ice-gpu kernel job: partition=ice-gpu, gres=gpu:h100:1,
# venv in repo, no module load needed (torch/vLLM bring their own CUDA).
#
# Usage (any #SBATCH can be overridden on the CLI; the CLI value wins):
#   sbatch scripts/slurm/gpu.sbatch python -u scripts/03_extract_activations.py
#   sbatch --job-name=lora --time=03:00:00 scripts/slurm/gpu.sbatch python -u scripts/07_train_lora.py
#   sbatch --time=00:30:00 scripts/slurm/gpu.sbatch python -u scripts/04_probe_and_monitor.py
#
# Env knobs:
#   COTMON_FS       scratch dir that holds HF_HOME (default: $HOME/scratch)
#   HF_HUB_OFFLINE  0 (default) — GT ICE compute nodes have internet, so a cache miss just
#                   downloads. Prestage still recommended to front-load ~50GB off GPU time.
set -eo pipefail
mkdir -p logs

COTMON_FS="${COTMON_FS:-$HOME/scratch}"
export HF_HOME="$COTMON_FS/hf"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

# load the same Python module the venv was built against (Lmod present on ICE nodes)
if type module >/dev/null 2>&1; then module load "${COTMON_PY_MODULE:-python/3.11.9}" 2>/dev/null || true; fi
source .venv/bin/activate    # built once via: source scripts/setup_box.sh "$COTMON_FS"

echo "[slurm] node=$(hostname)  job=${SLURM_JOB_ID:-?}  gpus=${CUDA_VISIBLE_DEVICES:-?}"
echo "[slurm] HF_HOME=$HF_HOME  HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$@"                          # e.g. python -u scripts/03_extract_activations.py
