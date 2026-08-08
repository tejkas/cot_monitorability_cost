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
from cotmon.probe import extract as ex  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default=str(config.DATA_DIR / "traces" / "phase2_tiers.jsonl"))
    ap.add_argument("--out-dir", default=str(config.ACTIVATIONS_DIR))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.in_path)]
    if args.limit:
        rows = rows[:args.limit]
    labels = np.array([r["label"] for r in rows])
    qids = np.array([r["question_id"] for r in rows])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    extractor = ex.ActivationExtractor()
    for tier in config.TIER_NAMES:
        texts = [r["tiers"][tier] for r in rows]
        acts = extractor.extract(texts)  # {pooling: [n, n_layers, hidden]}
        acts = {p: a.astype(config.ACT_DTYPE) for p, a in acts.items()}
        np.savez(out / f"acts_{tier}.npz", labels=labels, question_ids=qids, **acts)
        print(f"[phase3a] cached {tier}: " + ", ".join(f"{p}{tuple(a.shape)}" for p, a in acts.items()))


if __name__ == "__main__":
    main()
