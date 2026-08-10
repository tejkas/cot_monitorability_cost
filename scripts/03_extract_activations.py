#!/usr/bin/env python
"""Phase 3a — extract residual-stream activations for every tier (DESIGN §6).

Reads data/traces/phase2_tiers.jsonl, re-encodes each tier's TEXT through the base model (HF,
output_hidden_states), pools over tokens, and caches per-tier arrays to data/activations/ for
the probe sweep. GPU.

  python scripts/03_extract_activations.py [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402
from cotmon.data.generate import build_user_turn  # noqa: E402
from cotmon.data.hints import make_hint  # noqa: E402
from cotmon.probe import extract as ex  # noqa: E402


def load_mmlu_index() -> dict:
    """question_id -> {question, options} (trailing 'N/A' padding stripped, as in Phase 1)."""
    from datasets import load_dataset

    ds = load_dataset(config.PRIMARY_DATASET, split="test")
    idx = {}
    for i, r in enumerate(ds):
        opts = list(r["options"])
        while opts and str(opts[-1]).strip().upper() in {"N/A", ""}:
            opts.pop()
        idx[r.get("question_id", i)] = {"question": r["question"], "options": opts}
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default=str(config.DATA_DIR / "traces" / "phase2_tiers.jsonl"))
    ap.add_argument("--out-dir", default=str(config.ACTIVATIONS_DIR))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fresh", action="store_true",
                    help="re-extract even tiers whose .npz already exists (default: resume/skip them)")
    ap.add_argument("--no-prompt", action="store_true",
                    help="reproduce the old prompt-free extraction (for the methods comparison only)")
    ap.add_argument("--suffix", default=None,
                    help="output filename suffix; defaults to '_prompted' unless --no-prompt")
    args = ap.parse_args()
    suffix = args.suffix if args.suffix is not None else ("" if args.no_prompt else "_prompted")

    rows = [json.loads(l) for l in open(args.in_path)]
    if args.limit:
        rows = rows[:args.limit]
    labels = np.array([r["label"] for r in rows])
    qids = np.array([r["question_id"] for r in rows])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    todo = [t for t in config.TIER_NAMES
            if args.fresh or not (out / f"acts_{t}{suffix}.npz").exists()]
    done = [t for t in config.TIER_NAMES if t not in todo]
    if done:
        print(f"[phase3a] resume: skipping {done} (already cached; --fresh to redo)", flush=True)
    if not todo:
        print("[phase3a] all tiers already cached — nothing to do.")
        return

    extractor = ex.ActivationExtractor()

    # Rebuild the chat-templated prompt each trace was generated under, so the teacher-forced
    # pass conditions on the question AND the hint. Without this the probe is asked to detect
    # deference to a hint it never saw (and no paper extracts a response in isolation).
    prompts = None
    if not args.no_prompt:
        mmlu = load_mmlu_index()
        missing = [r["question_id"] for r in rows if r["question_id"] not in mmlu]
        if missing:
            raise SystemExit(f"[phase3a] {len(missing)} traces have no MMLU match; cannot build prompts")
        prompts = [extractor.chat_prompt(build_user_turn(
            mmlu[r["question_id"]]["question"], mmlu[r["question_id"]]["options"],
            make_hint(r["hint_family"], r["hint_target"]))) for r in rows]
        print(f"[phase3a] built {len(prompts)} chat prompts (question + hint in context)", flush=True)

    for tier in todo:
        texts = [r["tiers"][tier] for r in rows]
        n_empty = sum(1 for t in texts if not t or not t.strip())
        if n_empty:
            print(f"[phase3a] WARNING: {tier} has {n_empty} empty/blank texts (handled, but check Phase 2)",
                  flush=True)
        acts = extractor.extract(texts, prompts=prompts)  # {pooling: [n, n_layers, hidden]}
        acts = {p: a.astype(config.ACT_DTYPE) for p, a in acts.items()}
        np.savez(out / f"acts_{tier}{suffix}.npz", labels=labels, question_ids=qids, **acts)
        print(f"[phase3a] cached {tier}{suffix}: "
              + ", ".join(f"{p}{tuple(a.shape)}" for p, a in acts.items()), flush=True)


if __name__ == "__main__":
    main()
