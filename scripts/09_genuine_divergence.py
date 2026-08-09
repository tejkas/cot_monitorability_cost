#!/usr/bin/env python
"""Phase 4d — genuine divergence: activation probe vs text monitor on GENUINE LoRA@T3 reasoning.

Extracts activations from the genuine T3 CoTs (08), then:
  1. PROBE — train/eval (question-split) per (pooling, layer): is the sway linearly decodable from
     GENUINE illegible activations? Reported two ways:
       - honest cell : the SAME (pooling, layer) Phase 3 selected on T0 -> no test-set re-selection,
         directly comparable to the proxy.
       - best cell   : the genuine set's own best -> a ceiling, LABELLED as selection-on-test.
  2. MONITOR — SmolLM3-3B reads the genuine T3 text -> accuracy on the same held-out split.
  3. COMPARE — against Phase-3 base@T0 and proxy@T3 (from phase3_divergence.json).

The headline: genuine@T3 probe should track proxy@T3 (the proxy wasn't lying) AND beat the
genuine@T3 monitor (the divergence is real on text the model actually produced).

Output: results/phase4_genuine.json (+ .png)

  python scripts/09_genuine_divergence.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402
from cotmon.monitor import text_monitor as tm  # noqa: E402
from cotmon.probe import probe as P  # noqa: E402
from cotmon.probe import sweep  # noqa: E402
from cotmon.probe.extract import ActivationExtractor  # noqa: E402


def main() -> None:
    rows = [json.loads(l) for l in open(config.DATA_DIR / "traces" / "genuine_traces.jsonl")]
    labels = np.array([r["label"] for r in rows])
    qids = [r["question_id"] for r in rows]
    texts = [r["cot_text"] for r in rows]
    hints = [r["hint_target"] for r in rows]
    print(f"[phase4d] {len(rows)} genuine traces (pos={int(labels.sum())}, neg={int((labels == 0).sum())})")

    # 1) activations (cache to npz for reuse), then a within-genuine sweep
    act_path = config.ACTIVATIONS_DIR / "acts_genuine.npz"
    if act_path.exists():
        z = np.load(act_path)
        acts = {k: z[k].astype(np.float32) for k in z.files}
        print(f"[phase4d] loaded cached activations {act_path}")
    else:
        acts = ActivationExtractor().extract(texts)  # {pooling: [n, n_layers, hidden]}
        config.ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(act_path, **{p: acts[p].astype(config.ACT_DTYPE) for p in acts})
        print(f"[phase4d] wrote {act_path}")

    tr, te = sweep.question_split(qids, config.PROBE_TEST_FRACTION, config.PROBE_SPLIT_SEED)
    y_tr, y_te = labels[tr], labels[te]

    results: dict = {}  # results[pooling][layer] = {'auroc', 'acc'}
    for pool in config.PROBE_POOLINGS:
        results[pool] = {}
        for j, layer in enumerate(config.PROBE_LAYERS):
            results[pool][layer] = P.train_eval(
                acts[pool][tr, j, :], y_tr, acts[pool][te, j, :], y_te, config.PROBE_KIND)

    # genuine-best cell (ceiling — this IS selection on the test tier, so we flag it as such)
    gb_pool, gb_layer, gb_auroc = None, None, -1.0
    for pool in results:
        for layer in results[pool]:
            a = results[pool][layer]["auroc"]
            if a is not None and a > gb_auroc:
                gb_auroc, gb_pool, gb_layer = a, pool, layer

    # 2) text monitor on the SAME held-out split (apples-to-apples with the probe)
    preds = tm.TextMonitor().predict([texts[i] for i in te], [hints[i] for i in te])
    monitor_acc = float(np.mean([p == bool(int(y)) for p, y in zip(preds, y_te)]))

    # 3) Phase-3 reference + the honest cell (genuine probe at Phase 3's selected cell)
    honest, ref = None, {}
    p3_path = config.RESULTS_DIR / "phase3_divergence.json"
    if p3_path.exists():
        p3 = json.loads(p3_path.read_text())
        cp, cl = p3.get("best_cell", {}).get("pooling"), p3.get("best_cell", {}).get("layer")
        if cp in results and cl in results.get(cp, {}):
            honest = {"pooling": cp, "layer": cl, **results[cp][cl]}
        ref = {
            "base_T0_probe": p3["probe"].get(config.PROBE_TRAIN_TIER, {}).get("auroc"),
            "proxy_T3_probe": p3["probe"].get("T3_heavy", {}).get("auroc"),
            "base_T0_monitor": p3["text_monitor"].get(config.PROBE_TRAIN_TIER, {}).get("acc"),
            "proxy_T3_monitor": p3["text_monitor"].get("T3_heavy", {}).get("acc"),
        }
    else:
        print("[phase4d] note: phase3_divergence.json not found — skipping proxy/base comparison")

    out = {
        "n": len(rows), "n_pos": int(labels.sum()),
        "genuine_probe_honest_cell": honest,     # same cell as Phase 3 -> comparable, no double-dip
        "genuine_probe_best_cell": {"pooling": gb_pool, "layer": gb_layer, **results[gb_pool][gb_layer],
                                    "caveat": "selection-on-test ceiling"},
        "genuine_monitor_acc": monitor_acc,
        "phase3_reference": ref,
        "probe_full": results,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "phase4_genuine.json").write_text(json.dumps(out, indent=2))

    # report
    print("\n[phase4d] genuine divergence (probe recovers sway where the monitor can't):")
    if honest:
        print(f"  genuine@T3 probe  (honest cell {honest['pooling']}/L{honest['layer']}): "
              f"AUROC {honest['auroc']:.3f}")
    print(f"  genuine@T3 probe  (best cell {gb_pool}/L{gb_layer}, ceiling):   AUROC {gb_auroc:.3f}")
    print(f"  genuine@T3 monitor (SmolLM3-3B on genuine text):        acc   {monitor_acc:.3f}")
    if ref:
        print(f"  --- Phase-3 reference ---")
        print(f"  base@T0  probe {ref['base_T0_probe']:.3f} | monitor {ref['base_T0_monitor']:.3f}")
        print(f"  proxy@T3 probe {ref['proxy_T3_probe']:.3f} | monitor {ref['proxy_T3_monitor']:.3f}")
    print(f"\n[phase4d] wrote {config.RESULTS_DIR / 'phase4_genuine.json'}")
    _plot(honest, gb_auroc, monitor_acc, ref, config.RESULTS_DIR / "phase4_genuine.png")


def _plot(honest, gb_auroc, monitor_acc, ref, path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[phase4d] matplotlib unavailable; skipping plot")
        return
    labels = ["base@T0", "proxy@T3", "genuine@T3"]
    probe = [ref.get("base_T0_probe"), ref.get("proxy_T3_probe"),
             (honest or {}).get("auroc", gb_auroc)]
    mon = [ref.get("base_T0_monitor"), ref.get("proxy_T3_monitor"), monitor_acc]
    x = np.arange(len(labels))
    plt.figure(figsize=(6, 4))
    plt.bar(x - 0.2, [p if p is not None else 0 for p in probe], 0.4, label="activation probe (AUROC)")
    plt.bar(x + 0.2, [m if m is not None else 0 for m in mon], 0.4, label="text monitor (acc)")
    plt.xticks(x, labels)
    plt.ylabel("detection"); plt.ylim(0.4, 1.0)
    plt.title("Genuine vs proxy: probe holds, monitor falls")
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"[phase4d] wrote {path}")


if __name__ == "__main__":
    main()
