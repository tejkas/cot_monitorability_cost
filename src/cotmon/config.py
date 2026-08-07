"""Central config — settled decisions live here (see DESIGN.md §2).

Single source of truth for model IDs, dataset, tiers, and the probe sweep, so no
magic strings get scattered across phase scripts.
"""
from pathlib import Path

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
ACTIVATIONS_DIR = DATA_DIR / "activations"

# ---- Models ----
BASE_MODEL = "Qwen/Qwen3-8B"  # primary: toggleable thinking on ONE checkpoint (clean CoT-on/off delta)
BACKUP_MODELS = ["Qwen/Qwen3-4B-Thinking-2507", "HuggingFaceTB/SmolLM3-3B"]
# --- Phase 3 readers (NOT used in Phase 1/2; finalize in Phase 3) ---
# Two DISTINCT roles — do not collapse them (derived 2026-08-06):
#   TEXT_MONITOR: the reported monitor whose degradation across tiers IS the RQ1 result.
#     Must differ from REWRITER_MODEL (else it grades its own rewrites) and from BASE_MODEL
#     (same-model bias with the probed model). May be weak — its weakness is the phenomenon.
#   FAITHFULNESS_JUDGE: certifies tiers preserve the label via the T0-vs-T1 recovery gap.
#     Must be STRONG (near-ceiling on legible text, so a failure means the signal is truly
#     gone) and INDEPENDENT of the rewriter. Never a reported result -> may be as large as fits.
TEXT_MONITOR_MODEL = "HuggingFaceTB/SmolLM3-3B"  # DIFFERENT family from rewriter(4B) & base(8B) -> clean independence
FAITHFULNESS_JUDGE_MODEL = "Qwen/Qwen3-14B"      # stronger + independent of the 4B rewriter

# ---- Data (Phase 1) ----
PRIMARY_DATASET = "TIGER-Lab/MMLU-Pro"   # workhorse (difficulty-filtered in Phase 1) — verify ID on box
GENERALIZATION_EVAL = "Idavidrein/gpqa"  # GPQA-Diamond, OPTIONAL Phase-3 add-on (gated dataset)

# ---- Legibility tiers (Phase 2) ----
N_TIERS = 4  # T0 verbatim -> T3 near-unreadable
TIER_NAMES = ["T0_verbatim", "T1_mild", "T2_telegraphic", "T3_heavy"]
REWRITER_MODEL = "Qwen/Qwen2.5-14B-Instruct"  # SEPARATE family/gen from base(Qwen3-8B)+judge(Qwen3-14B); 4B over-summarized T1
REWRITE_TEMPERATURE = 0.3    # low -> faithful, consistent rewrites (not greedy: avoids loops)
REWRITE_TOP_P = 0.9
REWRITE_MAX_TOKENS = 8192    # T1 (clean prose) can approach input length; cap generously

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
GEN_MAX_NEW_TOKENS = 8192    # pilot: 4096 truncated ~29% of questions (0 valid samples) — recover the long tail
MAX_MODEL_LEN = 16384        # prompt + generation budget for the vLLM engine (KV-cache sizing)
SEED = 0

# Authoritative answer extraction (Phase 1): single-pass 'Final answer: (X)' + logprob check.
ANSWER_LOGPROBS = 5          # top-k logprobs at each token (5 covers the answer-letter margin; 20 was slow)
ANSWER_CONF_MARGIN = 0.20    # flag a trace 'flaky' if P(top letter) - P(runner-up) < this


def get_device() -> str:
    """cuda on the GPU box; mps/cpu for local code-path checks."""
    import torch  # lazy: keeps config importable on the Mac (no torch) for offline validation
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
