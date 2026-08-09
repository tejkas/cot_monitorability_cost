#!/usr/bin/env python
"""Phase 4a — build LoRA SFT pairs: (question prompt -> T3-style CoT + answer).

Joins phase2_tiers.jsonl (the T3 rewrites) with MMLU-Pro (question text/options by question_id)
and formats each as a chat example the model learns to imitate. Uses ONLY the TRAIN split
(question-level, same split as the probe) so the genuine-test questions stay unseen by the LoRA.

Output: data/traces/lora_sft.jsonl   (one {"messages": [...]} per line, trl-ready)

  python scripts/06_prep_lora_data.py
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
from cotmon.probe.sweep import question_split  # noqa: E402


def load_mmlu_index() -> dict:
    """question_id -> {question, options} (with trailing 'N/A' padding stripped, as in Phase 1)."""
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
    ap.add_argument("--out", default=str(config.DATA_DIR / "traces" / "lora_sft.jsonl"))
    ap.add_argument("--tier", default=config.LORA_TRAIN_TIER)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.in_path)]
    qids = np.array([r["question_id"] for r in rows])
    tr, _ = question_split(qids, config.PROBE_TEST_FRACTION, config.PROBE_SPLIT_SEED)
    tr_rows = set(tr.tolist())  # row indices in the TRAIN split
    mmlu = load_mmlu_index()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = miss = 0
    with out_path.open("w") as f:
        for i, r in enumerate(rows):
            if i not in tr_rows:
                continue
            q = mmlu.get(r["question_id"])
            if q is None:
                miss += 1
                continue
            prompt = build_user_turn(q["question"], q["options"],
                                     make_hint(r["hint_family"], r["hint_target"]))
            # target = genuine-looking assistant turn: T3-style thinking + the committed answer.
            target = f"<think>\n{r['tiers'][args.tier]}\n</think>\n\nFinal answer: ({r['answer']})"
            f.write(json.dumps({"messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ]}) + "\n")
            n += 1
    print(f"[phase4a] wrote {out_path}  ({n} SFT pairs from the train split; {miss} skipped: no MMLU match)")


if __name__ == "__main__":
    main()
