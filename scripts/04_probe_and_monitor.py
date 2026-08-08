#!/usr/bin/env python
"""Phase 3b — probe sweep + text monitor + faithfulness gate -> the RQ1 divergence (DESIGN §6-7).

Loads cached activations + tier text, then:
  1. probe sweep     -> probe AUROC per (pooling, layer, tier)   [needs probe/pooling bodies]
  2. text monitor    -> text-monitor accuracy per tier           [needs TextMonitor]
  3. faithfulness gate -> T0-vs-T1 recovery gap                  [needs FaithfulnessJudge]
  4. divergence      -> plot probe (best cell) vs text monitor across tiers

  python scripts/04_probe_and_monitor.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402
from cotmon.probe import sweep  # noqa: E402
from cotmon.monitor import text_monitor as tm  # noqa: E402
from cotmon.monitor import judge as jd  # noqa: E402


def load_activations(act_dir: Path):
    """acts_by_tier[tier][pooling] -> [n, n_layers, hidden]; plus labels, question_ids."""
    acts_by_tier, labels, qids = {}, None, None
    for tier in config.TIER_NAMES:
        z = np.load(act_dir / f"acts_{tier}.npz")
        acts_by_tier[tier] = {k: z[k] for k in z.files if k not in ("labels", "question_ids")}
        labels, qids = z["labels"], z["question_ids"]
    return acts_by_tier, labels, qids


def main() -> None:
    tiers_path = config.DATA_DIR / "traces" / "phase2_tiers.jsonl"
    rows = [json.loads(l) for l in open(tiers_path)]
    acts_by_tier, labels, qids = load_activations(config.ACTIVATIONS_DIR)

    # 1) probe sweep
    probe_results = sweep.run_sweep(acts_by_tier, labels, qids)

    # 2) text monitor per tier  --- Phase 3 ---
    monitor = tm.TextMonitor()
    hint_targets = [r["hint_target"] for r in rows]
    text_acc = {tier: None for tier in config.TIER_NAMES}  # TODO: monitor.predict per tier + score

    # 3) faithfulness gate (T0-vs-T1 recovery gap)  --- Phase 3 ---
    gate = jd.FaithfulnessJudge()
    faith = {"recovery": {}, "gap_T0_T1": None}  # TODO: gate.recover per tier + recovery_gap

    # 4) assemble + save + plot  --- Phase 3 ---
    out = config.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    with (out / "phase3_probe_results.json").open("w") as f:
        json.dump({"probe": probe_results, "text_acc": text_acc, "faithfulness": faith}, f, indent=2)
    print(f"[phase3b] wrote {out / 'phase3_probe_results.json'}")
    # TODO: divergence plot (probe best-cell vs text monitor across T0->T3)


if __name__ == "__main__":
    main()
