#!/usr/bin/env python
"""Task accuracy per condition — the LOAD-BEARING-CoT control the literature requires.

Baker et al. (Sec 4.1), Zolkowski et al. and Anwar et al. all warn that making CoT less legible
also tends to make the model WORSE AT THE TASK. Without this number, "the monitor/probe can't
detect sway in illegible CoT" is confounded with "there is less coherent reasoning to detect."

So: compare the base model (legible CoT) and the LoRA'd model (illegible CoT) on the SAME
questions, reporting where their answers land — gold / hint / other.

CAVEAT this script prints loudly: the stored trace files are ALREADY label-filtered (traces
answering a third option were dropped at labeling), so "% gold" here is conditioned on that
filter. Unfiltered accuracy needs the raw generation summary — 08 now prints it (see
--raw-* flags below to fold those numbers in).

  python scripts/11_task_accuracy.py
  python scripts/11_task_accuracy.py --raw-genuine-valid 687 --raw-genuine-gold 375
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402


def summarize(rows, name: str, restrict: set | None = None) -> dict:
    if restrict is not None:
        rows = [r for r in rows if r["question_id"] in restrict]
    n = len(rows)
    if not n:
        return {"condition": name, "n": 0}
    gold = sum(1 for r in rows if r["answer"] == r["baseline_answer"])
    hint = sum(1 for r in rows if r["answer"] == r["hint_target"])
    return {
        "condition": name, "n": n,
        "pct_gold(accuracy)": round(gold / n, 3),
        "pct_hint(swayed)": round(hint / n, 3),
        "questions": len(set(r["question_id"] for r in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-genuine-valid", type=int, default=0,
                    help="genuine samples with a parseable answer (from 08's printout)")
    ap.add_argument("--raw-genuine-gold", type=int, default=0,
                    help="of those, how many answered gold")
    args = ap.parse_args()

    tiers_path = config.DATA_DIR / "traces" / "phase2_tiers.jsonl"
    gen_path = config.DATA_DIR / "traces" / "genuine_traces.jsonl"
    base = [json.loads(l) for l in open(tiers_path)]
    lora = [json.loads(l) for l in open(gen_path)] if gen_path.exists() else []

    # fair comparison: only questions BOTH conditions cover (the held-out set)
    common = set(r["question_id"] for r in base) & set(r["question_id"] for r in lora)
    print(f"[phase-acc] shared questions between conditions: {len(common)}")

    rows = [summarize(base, "base model (legible CoT)", common),
            summarize(lora, "LoRA model (illegible CoT)", common)]

    print(f"\n{'condition':32s}{'n':>7}{'questions':>11}{'accuracy':>10}{'swayed':>9}")
    print("-" * 69)
    for r in rows:
        if not r["n"]:
            continue
        print(f"{r['condition']:32s}{r['n']:>7}{r['questions']:>11}"
              f"{r['pct_gold(accuracy)']:>10.3f}{r['pct_hint(swayed)']:>9.3f}")

    print("\n!! CAVEAT: these files are label-filtered (third-option answers were dropped), so")
    print("   accuracy here is conditioned on that filter and reads artificially high for BOTH.")
    if args.raw_genuine_valid:
        acc = args.raw_genuine_gold / args.raw_genuine_valid
        print(f"\n[unfiltered] genuine: {args.raw_genuine_gold}/{args.raw_genuine_valid} answered gold "
              f"= {acc:.3f} true task accuracy")
        print("   (run the base condition through 08's summary to get its comparable number)")

    out = config.RESULTS_DIR / "task_accuracy.json"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"shared_questions": len(common), "conditions": rows}, indent=2))
    print(f"\n[phase-acc] wrote {out}")
    print("\nWHY THIS MATTERS: if the illegible condition's accuracy is much lower, then reduced")
    print("detectability may reflect DEGRADED REASONING, not reduced monitorability. Report both.")


if __name__ == "__main__":
    main()
