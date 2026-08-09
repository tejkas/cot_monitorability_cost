#!/usr/bin/env bash
# cluster_recon.sh — run on the GT LOGIN node after cloning the repo.
# Read-only. Prints the facts needed to write correct SLURM + setup scripts.
# Usage:  bash scripts/cluster_recon.sh   (then paste the whole output back)

line() { printf '\n===== %s =====\n' "$1"; }

line "HOST / OS"
hostname; uname -a 2>/dev/null

line "SLURM PARTITIONS  (Partition | GRES/gpus | TimeLimit | #nodes | state)"
sinfo -o "%P %G %l %D %t" 2>/dev/null | head -60 || echo "sinfo not found"

line "GPU TYPES (unique GRES per partition)"
sinfo -o "%P %G" 2>/dev/null | sort -u

line "YOUR ACCOUNTS / QOS / PARTITIONS"
sacctmgr -s show user "$USER" format=User,Account,Partition,QOS%50 2>/dev/null | head -20 \
  || echo "sacctmgr not available"

line "MODULES: cuda / python / anaconda / gcc"
( module avail 2>&1 || true ) | tr '\t' '\n' | grep -iE "cuda|python|anaconda|miniconda|mamba|gcc" | sort -u | head -60

line "STORAGE"
echo "HOME=$HOME"; df -h "$HOME" 2>/dev/null
echo "SCRATCH env: SCRATCH=${SCRATCH:-<unset>}  TMPDIR=${TMPDIR:-<unset>}"
echo "scratch candidates:"; ls -ld ~/scratch /scratch/"$USER" "$SCRATCH" /storage/scratch1/"$USER" 2>/dev/null
echo "quota:"; quota -s 2>/dev/null | head -20 || echo "(quota cmd n/a)"

line "LOGIN-NODE INTERNET (can we reach HF Hub to pre-download models?)"
curl -sI --max-time 10 https://huggingface.co 2>&1 | head -1 || echo "curl failed / not installed"

line "NVIDIA DRIVER / CUDA (login node may have no GPU — that's fine)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
  || echo "no GPU on login node (expected)"

line "DONE"
echo "Next: compute-node internet must be tested under an allocation, e.g.:"
echo '  srun --partition=<P> --gres=gpu:1 --time=00:03:00 --pty bash -c \'
echo '    "hostname; curl -sI --max-time 10 https://huggingface.co | head -1 || echo NO-INTERNET"'
