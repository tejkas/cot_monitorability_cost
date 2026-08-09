#!/usr/bin/env python
"""Phase 3b — probe sweep + text monitor -> the RQ1 divergence (DESIGN §6-7).

Loads cached activations + tier text, then:
  1. probe sweep  -> AUROC per (pooling, layer, tier); pick the best cell on the TRAIN tier
  2. text monitor -> accuracy per tier (SmolLM3-3B reads the text), on the SAME held-out split
  3. divergence   -> probe (best cell) vs text monitor across T0->T3, saved + plotted

Faithfulness gate (Qwen3-14B) is a separate step (scripts/05) so only one model is in memory.

  python scripts/04_probe_and_monitor.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402
from cotmon.probe import probe as P  # noqa: E402
from cotmon.probe import sweep  # noqa: E402
from cotmon.monitor import text_monitor as tm  # noqa: E402


def load_activations(act_dir: Path):
    acts_by_tier, labels, qids = {}, None, None
    for tier in config.TIER_NAMES:
        z = np.load(act_dir / f"acts_{tier}.npz")
        acts_by_tier[tier] = {k: z[k] for k in z.files if k not in ("labels", "question_ids")}
        labels, qids = z["labels"], z["question_ids"]
    return acts_by_tier, labels, qids


def pick_best_cell(results: dict, tier: str):
    """Best (pooling, layer) by held-out AUROC on `tier` — model selection on the TRAIN tier only."""
    best, best_auroc = None, -1.0
    for pooling in results:
        for layer in results[pooling]:
            a = results[pooling][layer][tier]["auroc"]
            if a is not None and a > best_auroc:
                best_auroc, best = a, (pooling, layer)
    return best


def main() -> None:
    rows = [json.loads(l) for l in open(config.DATA_DIR / "traces" / "phase2_tiers.jsonl")]
    acts_by_tier, labels, qids = load_activations(config.ACTIVATIONS_DIR)

    # 1) probe sweep + best cell (selected on the train tier's held-out AUROC)
    probe_results = sweep.run_sweep(acts_by_tier, labels, qids)
    pool, layer = pick_best_cell(probe_results, config.PROBE_TRAIN_TIER)
    probe_by_tier = {t: probe_results[pool][layer][t] for t in config.TIER_NAMES}

    # 1b) within-tier DIAGONAL at the best cell: train on tier T, eval on tier T. Distinguishes
    # "signal is GONE from tier T" (diagonal ~chance too -> faithfulness/data problem) from "the
    # T0-trained probe just doesn't TRANSFER to T" (diagonal high, transfer low). Reads the T3 dip.
    tr, te = sweep.question_split(qids, config.PROBE_TEST_FRACTION, config.PROBE_SPLIT_SEED)
    j = config.PROBE_LAYERS.index(layer)
    diag_by_tier = {}
    for t in config.TIER_NAMES:
        X = acts_by_tier[t][pool]
        diag_by_tier[t] = P.train_eval(X[tr, j, :], labels[tr], X[te, j, :], labels[te], config.PROBE_KIND)

    # 2) text monitor on the SAME held-out split (leakage-safe, apples-to-apples). AUROC is the
    # headline (imbalance-robust); acc + yes_rate expose a degenerate constant classifier.
    y_te = labels[te]
    hints_te = [rows[i]["hint_target"] for i in te]
    base_rate = float(np.mean(y_te == 0))  # accuracy of always predicting "not swayed"
    monitor = tm.TextMonitor()
    text_by_tier = {}
    for t in config.TIER_NAMES:
        texts_te = [rows[i]["tiers"][t] for i in te]
        scores = np.array(monitor.predict_scores(texts_te, hints_te))
        preds = scores > 0.5
        text_by_tier[t] = {
            "auroc": P.auroc(y_te, scores),
            "acc": float(np.mean(preds == (y_te == 1))),
            "yes_rate": float(np.mean(preds)),
        }

    # 3) assemble + save + print the divergence
    out = config.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    result = {"best_cell": {"pooling": pool, "layer": layer},
              "base_rate": base_rate,
              "probe": probe_by_tier,            # T0-trained, evaluated on each tier (transfer)
              "probe_diagonal": diag_by_tier,    # trained AND evaluated within each tier
              "text_monitor": text_by_tier, "probe_full": probe_results}
    with (out / "phase3_divergence.json").open("w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[phase3] best probe cell: pooling={pool} layer={layer} (selected on {config.PROBE_TRAIN_TIER})")
    print(f"[phase3] base rate (always 'not swayed') = {base_rate:.3f}  <- monitor acc AT this = no signal")
    print(f"{'tier':16s}{'probe_xfer':>11s}{'probe_diag':>11s}{'mon_auroc':>10s}{'mon_acc':>9s}{'mon_yes%':>9s}")
    print("-" * 66)
    for t in config.TIER_NAMES:
        print(f"{t:16s}{probe_by_tier[t]['auroc']:>11.3f}{diag_by_tier[t]['auroc']:>11.3f}"
              f"{text_by_tier[t]['auroc']:>10.3f}{text_by_tier[t]['acc']:>9.3f}"
              f"{text_by_tier[t]['yes_rate']:>9.3f}")
    print(f"\n[phase3] wrote {out / 'phase3_divergence.json'}")

    _plot(probe_by_tier, text_by_tier, out / "phase3_divergence.png")


def _plot(probe_by_tier, text_by_tier, path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[phase3] matplotlib unavailable; skipping plot")
        return
    tiers = config.TIER_NAMES
    x = range(len(tiers))
    plt.figure(figsize=(6, 4))
    plt.plot(x, [probe_by_tier[t]["auroc"] for t in tiers], "o-", label="activation probe (AUROC)")
    plt.plot(x, [text_by_tier[t]["auroc"] for t in tiers], "s--", label="text monitor (AUROC)")
    plt.xticks(list(x), [t.split("_")[0] for t in tiers])
    plt.xlabel("legibility tier"); plt.ylabel("detection"); plt.ylim(0.4, 1.0)
    plt.title("Monitorability vs legibility"); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"[phase3] wrote {path}")


if __name__ == "__main__":
    main()
