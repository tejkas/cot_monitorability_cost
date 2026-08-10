#!/usr/bin/env python
"""Is the genuine 'no signal' a REAL negative or a pipeline BUG?

Three checks before anyone believes a literature-contradicting result:

1. POSITIVE CONTROL — train the same probe to decode the COMMITTED ANSWER LETTER from the same
   activations. The CoT literally ends with "Ans=F", so this MUST be decodable if the pipeline
   (extraction, row alignment, labels, pooling) is sound. A misalignment between activations and
   rows would kill this too. High AUROC => pipeline fine, the sway genuinely isn't linearly there.
   Chance AUROC => BUG; the negative result is meaningless.

2. CLUSTERED BOOTSTRAP CI — resample TEST QUESTIONS (not traces) with replacement. k=8 traces per
   item are highly correlated, so per-trace CIs are far too narrow and overstate our precision.

3. ACTIVATION HEALTH — norms/variance, and how many rows are near-duplicates (degenerate symbolic
   text could collapse the representation).

  python scripts/diag_probe_sanity.py                                  # genuine (base-model acts)
  python scripts/diag_probe_sanity.py --acts acts_genuine_loramodel.npz
  python scripts/diag_probe_sanity.py --proxy                          # same controls on proxy T0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402
from cotmon.probe import probe as P  # noqa: E402
from cotmon.probe import sweep  # noqa: E402

RNG = np.random.RandomState(0)


def best_cell_auroc(acts, y, tr, te, label):
    """Best (pooling, layer) AUROC + the scores at that cell."""
    best = (-1.0, None, None, None)
    for pool in config.PROBE_POOLINGS:
        for j, layer in enumerate(config.PROBE_LAYERS):
            m = P.train_eval(acts[pool][tr, j, :], y[tr], acts[pool][te, j, :], y[te], config.PROBE_KIND)
            if m["auroc"] is not None and m["auroc"] > best[0]:
                pr = P.DiffOfMeansProbe().fit(np.asarray(acts[pool][tr, j, :], dtype=np.float32), y[tr])
                s = pr.decision_scores(np.asarray(acts[pool][te, j, :], dtype=np.float32))
                best = (m["auroc"], pool, layer, s)
    print(f"  {label:34s} best AUROC {best[0]:.3f}  (cell {best[1]}/L{best[2]})")
    return best


def bootstrap_ci(y_te, scores, q_te, n_boot=2000):
    """95% CI resampling TEST QUESTIONS (clusters), not individual traces."""
    qs = np.array(q_te)
    uniq = np.unique(qs)
    idx_by_q = {q: np.where(qs == q)[0] for q in uniq}
    vals = []
    for _ in range(n_boot):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_q[q] for q in pick])
        yy, ss = y_te[rows], scores[rows]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(P.auroc(yy, ss))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="acts_genuine.npz")
    ap.add_argument("--proxy", action="store_true", help="run the controls on proxy T0 instead")
    args = ap.parse_args()

    if args.proxy:
        z = np.load(config.ACTIVATIONS_DIR / f"acts_{config.PROBE_TRAIN_TIER}.npz")
        acts = {p: z[p] for p in config.PROBE_POOLINGS}
        y_sway = z["labels"]
        qids = z["question_ids"]
        rows = [json.loads(l) for l in open(config.DATA_DIR / "traces" / "phase2_tiers.jsonl")]
        answers = [r["answer"] for r in rows]
        tag = "PROXY T0"
    else:
        z = np.load(config.ACTIVATIONS_DIR / args.acts)
        acts = {p: z[p] for p in config.PROBE_POOLINGS}
        rows = [json.loads(l) for l in open(config.DATA_DIR / "traces" / "genuine_traces.jsonl")]
        y_sway = np.array([r["label"] for r in rows])
        qids = np.array([r["question_id"] for r in rows])
        answers = [r["answer"] for r in rows]
        tag = f"GENUINE ({args.acts})"

    n = len(y_sway)
    assert acts[config.PROBE_POOLINGS[0]].shape[0] == n == len(answers), \
        f"ROW MISMATCH: acts={acts[config.PROBE_POOLINGS[0]].shape[0]} labels={n} rows={len(answers)}"
    tr, te = sweep.question_split(qids, config.PROBE_TEST_FRACTION, config.PROBE_SPLIT_SEED)

    print(f"\n===== {tag} =====")
    print(f"n={n}  pos={int(y_sway.sum())}  unique questions={len(np.unique(qids))}")
    print(f"test: n={len(te)} pos={int(y_sway[te].sum())} "
          f"questions={len(np.unique(qids[te]))} questions_with_pos="
          f"{len(np.unique(qids[te][y_sway[te] == 1]))}   <- effective sample size")

    # --- 1a. ALIGNMENT CONTROL: decode TEXT LENGTH. Pooled activations must encode this, and it
    # breaks instantly if activation rows are misaligned with the jsonl rows. This is the
    # unambiguous bug test (the answer-letter control below may just be too fine-grained).
    texts = [r["tiers"][config.PROBE_TRAIN_TIER] if args.proxy else r["cot_text"] for r in rows]
    lens = np.array([len(t) for t in texts])
    y_len = (lens > np.median(lens)).astype(int)
    print(f"\n[1a] ALIGNMENT CONTROL — decode 'text longer than median' (must be easy; "
          f"fails if rows are misaligned)")
    ctrl_len = best_cell_auroc(acts, y_len, tr, te, "text-length > median")

    # --- 1b. POSITIVE CONTROL: decode the committed answer letter ---
    from collections import Counter
    top_letter = Counter(a for a in answers if a).most_common(1)[0][0]
    y_letter = np.array([1 if a == top_letter else 0 for a in answers])
    print(f"\n[1b] POSITIVE CONTROL — decode 'answer == {top_letter}' "
          f"(pos={int(y_letter.sum())}/{n}); the CoT states it, but it is ONE token in ~200 "
          f"so pooling may legitimately wash it out")
    ctrl = best_cell_auroc(acts, y_letter, tr, te, f"answer-letter=={top_letter}")

    # --- 2. the sway target, with a CLUSTERED CI ---
    print(f"\n[2] SWAY TARGET (the actual result), with question-clustered 95% CI")
    sway = best_cell_auroc(acts, y_sway, tr, te, "swayed?")
    lo, hi = bootstrap_ci(y_sway[te], sway[3], qids[te])
    print(f"  {'':34s} 95% CI [{lo:.3f}, {hi:.3f}]  (resampling QUESTIONS, not traces)")

    # --- 3. activation health ---
    A = np.asarray(acts["mean"][:, len(config.PROBE_LAYERS) // 2, :], dtype=np.float32)
    norms = np.linalg.norm(A, axis=1)
    print(f"\n[3] ACTIVATION HEALTH (mean-pool, mid layer): norm med={np.median(norms):.1f} "
          f"min={norms.min():.1f} max={norms.max():.1f}; "
          f"per-dim std med={np.median(A.std(axis=0)):.4f}; "
          f"non-finite={int((~np.isfinite(A)).sum())}")

    print("\n===== VERDICT =====")
    if ctrl_len[0] < 0.70:
        print(f"  ALIGNMENT control FAILED (length AUROC {ctrl_len[0]:.3f}) -> activations do not "
              "even track text length.")
        print("  => rows are misaligned or extraction is broken; this result is VOID.")
    else:
        print(f"  ALIGNMENT control PASSED (length AUROC {ctrl_len[0]:.3f}) -> rows line up, "
              "extraction works.")
        print(f"  (answer-letter control {ctrl[0]:.3f} — informative only if it PASSES on the proxy "
              "too; if it fails on both, it is simply too fine-grained for pooled activations.)")
        print(f"  => sway result stands as measured: {sway[0]:.3f}, CI [{lo:.3f}, {hi:.3f}].")
        print("     Report the INTERVAL. If it spans the proxy value, the genuine arm is "
              "INCONCLUSIVE, not a contradiction.")


if __name__ == "__main__":
    main()
