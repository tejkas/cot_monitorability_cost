"""Central config — settled decisions live here (see DESIGN.md §2).

Single source of truth for model IDs, dataset, tiers, and the probe sweep, so no
magic strings get scattered across phase scripts.
"""
from pathlib import Path

import torch

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
ACTIVATIONS_DIR = DATA_DIR / "activations"

# ---- Models ----
BASE_MODEL = "Qwen/Qwen3-8B"  # primary: toggleable thinking on ONE checkpoint (clean CoT-on/off delta)
BACKUP_MODELS = ["Qwen/Qwen3-4B-Thinking-2507", "HuggingFaceTB/SmolLM3-3B"]
# Text-monitor judge (Phase 3): a DIFFERENT small model, to avoid same-model bias.
JUDGE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # placeholder; revisit in Phase 3

# ---- Data (Phase 1) ----
PRIMARY_DATASET = "TIGER-Lab/MMLU-Pro"   # workhorse (difficulty-filtered in Phase 1) — verify ID on box
GENERALIZATION_EVAL = "Idavidrein/gpqa"  # GPQA-Diamond, OPTIONAL Phase-3 add-on (gated dataset)

# ---- Legibility tiers (Phase 2) ----
N_TIERS = 4  # T0 verbatim -> T3 near-unreadable
TIER_NAMES = ["T0_verbatim", "T1_mild", "T2_telegraphic", "T3_heavy"]

# ---- Probe sweep (Phase 3) ----
# Qwen3-8B has 36 transformer blocks; sweep ~every 4th layer.
PROBE_LAYERS = list(range(4, 36, 4))  # [4, 8, 12, 16, 20, 24, 28, 32]
PROBE_POOLINGS = ["mean", "max", "last", "softmax", "attention"]  # McKenzie et al. Table 3

# ---- Phase 1: hint-usage data generation ----
# Labeling scheme derived in DESIGN.md §4. Pilot defaults; tune after the first run.
N_QUESTIONS = 300            # candidate questions to screen (pilot size)
K_SAMPLES = 8               # generations per (question, condition) to estimate P0 / P1
P0_STABILITY = 0.75         # min P0(mode) to call the unhinted baseline "peaked/confident"
REQUIRE_BASELINE_CORRECT = True  # keep only questions where a0* == gold ("capable")
LIFT_THRESHOLD = 0.5        # P1(h)-P0(h) that marks a "flip question" (stored; NOT a hard gate)
HINT_FAMILIES = ["sycophancy", "authority"]

# Qwen3 thinking-mode sampling (from the model card).
GEN_TEMPERATURE = 0.6
GEN_TOP_P = 0.95
GEN_TOP_K = 20
GEN_MAX_NEW_TOKENS = 4096    # thinking traces on MMLU-Pro run long; 2048 truncated a biased tail
MAX_MODEL_LEN = 8192         # prompt + generation budget for the vLLM engine (KV-cache sizing)
SEED = 0


def get_device() -> str:
    """cuda on the GPU box; mps/cpu for local code-path checks."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
