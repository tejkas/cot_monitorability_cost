#!/usr/bin/env python
"""Phase 2 — rewrite T0 CoTs into legibility tiers T1..T3 and validate the gradient (DESIGN §5).

A SEPARATE instruct model (config.REWRITER_MODEL) rewrites each Phase-1 trace's CoT with a
FROZEN prompt per tier — no optimization against any monitor, so we degrade legibility without
inducing adversarial obfuscation. T0 is the verbatim original.

Input:  data/traces/phase1_traces.jsonl
Outputs:
  data/traces/phase2_tiers.jsonl        one row per trace: carried label/meta + tiers{T0..T3}
  results/phase2_tier_validation.json   per-tier readability / keyword-retention means

Run on the GPU box:
  python scripts/02_make_tiers.py
Quick check first (rewrites only 20 traces):
  python scripts/02_make_tiers.py --limit 20
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402
from cotmon.tiers import rewriter as rw  # noqa: E402
from cotmon.tiers import validate as val  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default=str(config.DATA_DIR / "traces" / "phase1_traces.jsonl"))
    ap.add_argument("--out", default=str(config.DATA_DIR / "traces" / "phase2_tiers.jsonl"))
    ap.add_argument("--model", default=config.REWRITER_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="rewrite only the first N traces (0 = all)")
    args = ap.parse_args()

    traces = [json.loads(l) for l in open(args.in_path)]
    if args.limit:
        traces = traces[:args.limit]
    cots = [t["cot_text"] for t in traces]
    print(f"[phase2] rewriting {len(cots)} traces x {len(rw.REWRITE_TIERS)} tiers with {args.model}")

    R = rw.Rewriter(model=args.model)
    tier_sets = R.rewrite(cots)  # list of {tier_name: text}

    # --- write the tiered dataset (carry the label + provenance through) ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    carry = ["question_id", "hint_family", "hint_target", "baseline_answer",
             "answer", "label", "lift", "flip_question"]
    with out_path.open("w") as f:
        for t, tiers in zip(traces, tier_sets):
            row = {k: t.get(k) for k in carry}
            row["tiers"] = tiers
            f.write(json.dumps(row) + "\n")
    print(f"[phase2] wrote {out_path}  ({len(tier_sets)} rows)")

    # --- validation: per-tier metric means + monotonicity verdict ---
    summary = val.summarize(tier_sets)
    res_path = config.RESULTS_DIR / "phase2_tier_validation.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with res_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[phase2] tier validation (means across {summary['n']} traces):")
    hdr = f"{'tier':16s} {'readability':>11s} {'fw_ratio':>9s} {'symbol':>7s} {'len_chars':>9s} {'kw_retain':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for tier in config.TIER_NAMES:
        m = summary["per_tier"][tier]
        print(f"{tier:16s} {m['readability']:>11.3f} {m['fw_ratio']:>9.3f} "
              f"{m['symbol_ratio']:>7.3f} {m['len_chars']:>9.0f} {m['keyword_retention']:>9.3f}")
    print(f"\n[phase2] readability decreasing T0->T3:        {summary['readability_monotone']}")
    print(f"[phase2] keyword-retention decreasing T0->T3:  {summary['keyword_monotone']}")
    print(f"[phase2] wrote {res_path}")


if __name__ == "__main__":
    main()
