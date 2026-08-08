#!/usr/bin/env python
"""Phase 3c — faithfulness gate: did the sway survive each tier? (DESIGN §6 / faithfulness notes)

A strong independent judge (Qwen3-14B) recovers "swayed?" from each tier's text on the SAME
held-out split; the T0-vs-T1 recovery GAP flags rewriter deletion — T1 is legible, so a drop
there is lost signal, not illegibility. Separate from script 04 so only one model is in GPU
memory at a time.

  python scripts/05_faithfulness_gate.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402
from cotmon.probe import sweep  # noqa: E402
from cotmon.monitor import judge as jd  # noqa: E402


def main() -> None:
    rows = [json.loads(l) for l in open(config.DATA_DIR / "traces" / "phase2_tiers.jsonl")]
    labels = np.array([r["label"] for r in rows])
    qids = np.array([r["question_id"] for r in rows])
    _, te = sweep.question_split(qids, config.PROBE_TEST_FRACTION, config.PROBE_SPLIT_SEED)
    y_te = labels[te]
    hints_te = [rows[i]["hint_target"] for i in te]

    gate = jd.FaithfulnessJudge()
    recovery = {}
    for t in config.TIER_NAMES:
        texts_te = [rows[i]["tiers"][t] for i in te]
        recovery[t] = gate.recover(texts_te, hints_te, y_te)

    gap = jd.recovery_gap(recovery["T0_verbatim"], recovery["T1_mild"])
    verdict = "FAITHFUL" if gap <= config.FAITH_MAX_GAP else "UNFAITHFUL — rewriter dropping sway at T1"

    out = config.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    res = {"recovery": recovery, "gap_T0_T1": gap, "max_gap": config.FAITH_MAX_GAP, "verdict": verdict}
    with (out / "phase3_faithfulness.json").open("w") as f:
        json.dump(res, f, indent=2)

    print("[phase3] judge recovery of the sway, per tier (on the held-out positives):")
    for t in config.TIER_NAMES:
        print(f"  {t:16s} {recovery[t]:.3f}")
    print(f"[phase3] T0->T1 recovery gap = {gap:.3f} (threshold {config.FAITH_MAX_GAP}) -> {verdict}")
    print(f"[phase3] wrote {out / 'phase3_faithfulness.json'}")


if __name__ == "__main__":
    main()
