#!/usr/bin/env bash
# setup_box.sh — one-command environment setup for the Lambda GPU box.
#
# SOURCE this (do not just execute it) so the venv activation and HF_HOME
# persist into your current shell:
#
#     source scripts/setup_box.sh /home/ubuntu/<filesystem>   # persistent-storage mount
#     source scripts/setup_box.sh                             # uses $COTMON_FS, else $HOME (no persistence)
#     source scripts/setup_box.sh <fs> --fresh                # rebuild the venv from scratch
#
# Re-run on every fresh spin. It is idempotent: if the venv + deps already exist
# on the (persistent) filesystem it just re-activates them in seconds. Use --fresh
# if you switched instance types and the CUDA driver changed under the compiled
# torch / vLLM wheels.
#
# What it does: points HF_HOME at the persistent filesystem (so Qwen3-8B's ~16 GB
# download happens ONCE, not per spin), builds/activates .venv, installs deps in
# the vLLM-first order that avoids the torch pin conflict, and verifies GPU/torch/vllm.

_cotmon_setup() {
  local src repo_root fresh=0 fs="" a
  src="${BASH_SOURCE[0]:-$0}"
  repo_root="$(cd "$(dirname "$src")/.." && pwd)"

  for a in "$@"; do
    case "$a" in
      --fresh) fresh=1 ;;
      -*)      echo "[setup] ignoring unknown flag: $a" ;;
      *)       fs="$a" ;;
    esac
  done
  if [ -z "$fs" ]; then
    fs="${COTMON_FS:-$HOME}"
    [ -z "$COTMON_FS" ] && echo "[setup] WARN: no persistent filesystem given → HF cache under \$HOME will NOT survive termination. Pass the mount path (or set COTMON_FS)."
  fi

  echo "[setup] repo:        $repo_root"
  echo "[setup] filesystem:  $fs"

  # --- HF cache on the persistent filesystem (download the model ONCE) ---
  export HF_HOME="$fs/hf"
  if ! mkdir -p "$HF_HOME"; then echo "[setup] ERROR: cannot create $HF_HOME"; return 1; fi
  echo "[setup] HF_HOME=$HF_HOME"

  cd "$repo_root" || return 1

  # --- venv (lives in the repo; persists if the repo is on the filesystem) ---
  if [ "$fresh" = 1 ] && [ -d .venv ]; then
    echo "[setup] --fresh: removing existing .venv"
    rm -rf .venv
  fi
  if [ ! -d .venv ]; then
    echo "[setup] creating venv (.venv) ..."
    python3 -m venv .venv || { echo "[setup] ERROR: venv creation failed (need python3-venv?)"; return 1; }
  fi
  # shellcheck source=/dev/null
  source .venv/bin/activate || { echo "[setup] ERROR: could not activate .venv"; return 1; }

  # --- deps: skip if already installed unless --fresh ---
  if [ "$fresh" = 1 ] || [ ! -f .venv/.deps_installed ]; then
    echo "[setup] installing deps (slow: vLLM + torch) ..."
    python -m pip install --upgrade pip wheel || return 1
    # vLLM pins its own torch build. Install it FIRST so torch resolves to the
    # version vLLM needs; requirements.txt (torch>=2.4) is then already satisfied
    # and pip won't try to bump torch and start a conflict.
    pip install vllm || { echo "[setup] ERROR: vllm install failed"; return 1; }
    pip install -r requirements.txt || { echo "[setup] ERROR: requirements install failed"; return 1; }
    touch .venv/.deps_installed
    echo "[setup] deps installed."
  else
    echo "[setup] deps already present (pass --fresh to rebuild)."
  fi

  # --- verify (guarded so a partial env can't kill a sourced shell) ---
  echo "[setup] ---- verification ----"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  else
    echo "[setup] WARN: nvidia-smi not found — is this actually the GPU box?"
  fi
  python - <<'PY'
try:
    import torch
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
except Exception as e:
    print("torch import FAILED:", e)
try:
    import vllm
    print("vllm ", vllm.__version__)
except Exception as e:
    print("vllm import FAILED:", e)
PY

  echo "[setup] ready. venv active, HF_HOME set, cwd=$repo_root"
  echo "[setup] next:  python scripts/00_smoke_test.py --model Qwen/Qwen3-0.6B"
  echo "[setup]        python scripts/01_generate_traces.py --n-questions 300 --k 8"
  return 0
}

_cotmon_setup "$@"
