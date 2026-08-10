#!/usr/bin/env python
"""Diagnostic B (GPU): does the T3 LoRA generalize, or did it overfit?

Generate GREEDILY with the adapter on a few TRAIN-split questions and a few HELD-OUT questions,
and compare the style (length). Verdict:
  TRAIN short/symbolic + TEST long/verbose -> OVERFITTING (memorized train, no generalization)
                                              => fix = MORE DATA / regularization / fewer epochs
  BOTH verbose                             -> never learned a general transform
                                              => fix = more data + capacity (or approach is too weak)
  BOTH short/symbolic                      -> it DOES generalize (then 08's verbose is a config bug)

  python scripts/diag_generalization.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402
from cotmon.data import generate as gen  # noqa: E402
from cotmon.data.hints import make_hint  # noqa: E402
from cotmon.probe.sweep import question_split  # noqa: E402


def load_mmlu_index() -> dict:
    from datasets import load_dataset

    ds = load_dataset(config.PRIMARY_DATASET, split="test")
    idx = {}
    for i, r in enumerate(ds):
        opts = list(r["options"])
        while opts and str(opts[-1]).strip().upper() in {"N/A", ""}:
            opts.pop()
        idx[r.get("question_id", i)] = {"question": r["question"], "options": opts}
    return idx


def pick(rows, idxset, n):
    seen, out = set(), []
    for i, r in enumerate(rows):
        if i not in idxset:
            continue
        k = (r["question_id"], r["hint_family"], r["hint_target"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
        if len(out) >= n:
            break
    return out


def main() -> None:
    rows = [json.loads(l) for l in open(config.DATA_DIR / "traces" / "phase2_tiers.jsonl")]
    qids = np.array([r["question_id"] for r in rows])
    tr, te = question_split(qids, config.PROBE_TEST_FRACTION, config.PROBE_SPLIT_SEED)
    tagged = ([("TRAIN", r) for r in pick(rows, set(tr.tolist()), 4)]
              + [("TEST ", r) for r in pick(rows, set(te.tolist()), 4)])

    mmlu = load_mmlu_index()
    turns, nopts = [], []
    for _tag, r in tagged:
        q = mmlu[r["question_id"]]
        turns.append(gen.build_user_turn(q["question"], q["options"],
                                         make_hint(r["hint_family"], r["hint_target"])))
        nopts.append(len(q["options"]))

    G = gen.Generator(model=config.BASE_MODEL, lora=str(config.RESULTS_DIR / config.LORA_OUT))
    samples = G.sample(turns, 1, nopts, temperature=0.0)  # greedy

    print("\n===== TRAIN vs HELD-OUT greedy generations (with the LoRA) =====")
    for (tag, r), samps in zip(tagged, samples):
        cot = samps[0].cot
        style = "SYMBOLIC" if len(cot) < 1000 else "verbose"
        print(f"[{tag}] qid={r['question_id']}  len={len(cot):>6} chars  -> {style}")
        print("    " + cot[:220].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
