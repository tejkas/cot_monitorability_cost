#!/usr/bin/env python
"""Expand the question pool: capability screen + CoT-necessity screen (base model, unhinted).

Two filters, both required for a valid sway experiment:

1. CAPABLE — unhinted modal answer == gold AND peaked (>= P0_STABILITY). The hint always points
   at a WRONG option, so a flip to the hint is attributable to the hint only if the model
   reliably knew the answer on its own. Without this, "swayed" and "ignorant" are the same event.

2. CoT-NECESSARY — accuracy with thinking must beat accuracy without it. The literature admits a
   task only if reasoning measurably helps (Zolkowski screens on a significant CoT-vs-no-CoT gain;
   Anwar runs the same check post hoc and honestly reports the tasks that fail it). Otherwise
   "degrading the CoT hurts monitorability" is confounded with "the CoT never mattered".
   Qwen3 toggles thinking on the SAME weights, so this is a clean within-model contrast.

Questions already used in Phases 1-2 are excluded, so the new pool is disjoint from the LoRA's
SFT questions (Set A) and can be split into probe-train (B) and evaluation (C).

  python scripts/12_screen_questions.py --n-questions 800 --k 8
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402
from cotmon.data import generate as gen  # noqa: E402
from cotmon.data import labeling as lab  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=800)
    ap.add_argument("--k", type=int, default=config.K_SAMPLES)
    ap.add_argument("--exclude", default=str(config.DATA_DIR / "traces" / "phase2_tiers.jsonl"),
                    help="traces whose question_ids are already used (Set A / old pool)")
    ap.add_argument("--out", default=str(config.DATA_DIR / "traces" / "screened_questions.jsonl"))
    ap.add_argument("--no-think-tokens", type=int, default=512,
                    help="token cap for the no-CoT arm (it should answer directly)")
    args = ap.parse_args()

    from datasets import load_dataset

    used = set()
    if Path(args.exclude).exists():
        used = {json.loads(l)["question_id"] for l in open(args.exclude)}
    print(f"[screen] excluding {len(used)} question_ids already used")

    ds = load_dataset(config.PRIMARY_DATASET, split="test")
    cand = []
    for i, r in enumerate(ds):
        qid = r.get("question_id", i)
        if qid in used:
            continue
        opts = list(r["options"])
        while opts and str(opts[-1]).strip().upper() in {"N/A", ""}:
            opts.pop()
        if len(opts) < 4:
            continue
        cand.append({"question_id": qid, "question": r["question"], "options": opts,
                     "gold": r["answer"]})
        if len(cand) >= args.n_questions:
            break
    print(f"[screen] {len(cand)} candidate questions")

    turns = [gen.build_user_turn(c["question"], c["options"]) for c in cand]
    nopts = [len(c["options"]) for c in cand]
    G = gen.Generator(model=config.BASE_MODEL)

    # --- pass 1: WITH thinking -> capability filter + the CoT arm of the necessity check ---
    print("[screen] pass 1/2: unhinted WITH thinking", flush=True)
    think = G.sample(turns, args.k, nopts, thinking=True)

    # --- pass 2: WITHOUT thinking -> the no-CoT arm (short cap; it should answer directly) ---
    print("[screen] pass 2/2: unhinted WITHOUT thinking", flush=True)
    nothink = G.sample(turns, args.k, nopts, thinking=False, max_tokens=args.no_think_tokens)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_cap = n_both = 0
    with out_path.open("w") as f:
        for c, s_t, s_n in zip(cand, think, nothink):
            p0, n_valid = lab.distribution(s_t, len(c["options"]))
            capable = lab.is_capable(p0, c["gold"], n_valid, min_valid=max(2, args.k // 2))
            acc_think = sum(1 for s in s_t if s.answer == c["gold"]) / max(len(s_t), 1)
            acc_nothink = sum(1 for s in s_n if s.answer == c["gold"]) / max(len(s_n), 1)
            rec = {**{k: c[k] for k in ("question_id", "gold")},
                   "capable": bool(capable), "n_valid": n_valid,
                   "p0_peak": max(p0.values()) if p0 else 0.0,
                   "acc_think": acc_think, "acc_nothink": acc_nothink,
                   "cot_gain": acc_think - acc_nothink}
            f.write(json.dumps(rec) + "\n")
            n_cap += int(capable)
            n_both += int(capable and rec["cot_gain"] > 0)

    print(f"\n[screen] wrote {out_path}")
    print(f"[screen] capable: {n_cap}/{len(cand)} ({n_cap / max(len(cand), 1):.1%})")
    print(f"[screen] capable AND CoT-positive (cot_gain>0): {n_both}")
    print("[screen] NOTE: report the pooled CoT-vs-no-CoT accuracy gap in the writeup — it is the "
          "load-bearing-CoT control the literature requires.")


if __name__ == "__main__":
    main()
