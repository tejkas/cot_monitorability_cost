#!/usr/bin/env python
"""Phase 4 (expanded) — generate BOTH conditions on the same questions, with a clean B/C split.

Conditions, same questions and same hints throughout:
  LEGIBLE   = base Qwen3-8B (verbose CoT)
  ILLEGIBLE = the T3 LoRA (symbolic CoT)

One vLLM engine serves both (enable_lora + per-request adapter), so the two conditions differ only
in the adapter — not in prompts, sampling code, or engine version.

Question sets (each used exactly once, no question appears in two roles):
  Set A = the LoRA's SFT questions (the OLD pool) -> excluded here entirely
  Set B = 40% of newly screened questions -> probe TRAINING
  Set C = 60% -> EVALUATION
Eval-heavy because underpowered evaluation is what made the first genuine run inconclusive
(12 positive-bearing eval questions); probe training was never the bottleneck.

Input:  data/traces/screened_questions.jsonl  (from 12)
Output: data/traces/conditions.jsonl          (one labeled trace per valid sample)

  python scripts/13_generate_conditions.py --k 8
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402
from cotmon.data import generate as gen  # noqa: E402
from cotmon.data import labeling as lab  # noqa: E402
from cotmon.data.hints import make_hint  # noqa: E402

LETTERS = gen.LETTERS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=config.K_SAMPLES)
    ap.add_argument("--screened", default=str(config.DATA_DIR / "traces" / "screened_questions.jsonl"))
    ap.add_argument("--out", default=str(config.DATA_DIR / "traces" / "conditions.jsonl"))
    ap.add_argument("--lora", default=str(config.RESULTS_DIR / config.LORA_OUT))
    ap.add_argument("--eval-frac", type=float, default=0.60, help="fraction of questions in Set C")
    ap.add_argument("--authority-frac", type=float, default=0.80,
                    help="share of questions given an authority hint (it flips ~41% vs sycophancy ~3%)")
    ap.add_argument("--require-cot-gain", action="store_true",
                    help="keep only questions where thinking beat no-thinking (CoT-necessity screen)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    screened = [json.loads(l) for l in open(args.screened)]
    keep = [s for s in screened if s["capable"] and (not args.require_cot_gain or s["cot_gain"] > 0)]
    print(f"[phase4x] screened={len(screened)}  capable={sum(s['capable'] for s in screened)}  "
          f"kept={len(keep)}" + ("  (CoT-necessity enforced)" if args.require_cot_gain else ""))
    if args.limit:
        keep = keep[:args.limit]

    # question text/options
    ds = load_dataset(config.PRIMARY_DATASET, split="test")
    qinfo = {}
    for i, r in enumerate(ds):
        qid = r.get("question_id", i)
        opts = list(r["options"])
        while opts and str(opts[-1]).strip().upper() in {"N/A", ""}:
            opts.pop()
        qinfo[qid] = {"question": r["question"], "options": opts}

    # --- B/C split by QUESTION, plus hint assignment (deterministic) ---
    rng = random.Random(config.SEED)
    qids = [s["question_id"] for s in keep]
    rng.shuffle(qids)
    n_eval = int(round(len(qids) * args.eval_frac))
    set_c = set(qids[:n_eval])          # evaluation
    items = []
    for s in keep:
        q = qinfo[s["question_id"]]
        wrong = [L for L in LETTERS[:len(q["options"])] if L != s["gold"]]
        if not wrong:
            continue
        fam = "authority" if rng.random() < args.authority_frac else "sycophancy"
        items.append({"question_id": s["question_id"], "gold": s["gold"],
                      "hint_family": fam, "hint_target": rng.choice(wrong),
                      "set": "C" if s["question_id"] in set_c else "B",
                      "question": q["question"], "options": q["options"]})
    print(f"[phase4x] {len(items)} items -> Set B (probe train) {sum(i['set'] == 'B' for i in items)}, "
          f"Set C (eval) {sum(i['set'] == 'C' for i in items)}")
    print(f"[phase4x] hint families: {dict(Counter(i['hint_family'] for i in items))}")

    turns = [gen.build_user_turn(i["question"], i["options"],
                                 make_hint(i["hint_family"], i["hint_target"])) for i in items]
    nopts = [len(i["options"]) for i in items]

    G = gen.Generator(model=config.BASE_MODEL, lora=args.lora)   # one engine, both conditions

    runs = [
        # (name, use_lora, temperature, repetition_penalty)
        ("legible", False, config.GEN_TEMPERATURE, 1.0),
        ("illegible", True, config.GEN_TEMPERATURE, 1.3),   # penalty tames the symbolic-loop failure
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    with out_path.open("w") as f:
        for name, use_lora, temp, rep in runs:
            print(f"\n[phase4x] generating condition '{name}' (lora={use_lora}, temp={temp}, "
                  f"rep_penalty={rep})", flush=True)
            samples = G.sample(turns, args.k, nopts, temperature=temp,
                               repetition_penalty=rep, use_lora=use_lora)

            # UNFILTERED answer distribution = the load-bearing-CoT control. Must be computed
            # before labeling, which drops third-option answers and inflates apparent accuracy.
            dist = {"gold": 0, "hint": 0, "other": 0, "none": 0}
            for it, samps in zip(items, samples):
                for s in samps:
                    if s.answer is None:
                        dist["none"] += 1
                    elif s.answer == it["gold"]:
                        dist["gold"] += 1
                    elif s.answer == it["hint_target"]:
                        dist["hint"] += 1
                    else:
                        dist["other"] += 1
            valid = dist["gold"] + dist["hint"] + dist["other"]
            acc = dist["gold"] / valid if valid else float("nan")

            n_pos = n_neg = 0
            for it, samps in zip(items, samples):
                for rec in lab.label_traces(it["question_id"], it["hint_family"], it["hint_target"],
                                            it["gold"], {}, {}, samps):
                    rec["condition"] = name
                    rec["set"] = it["set"]
                    f.write(json.dumps(rec) + "\n")
                    n_pos += int(rec["label"] == 1)
                    n_neg += int(rec["label"] == 0)
            summary[name] = {"answers": dist, "task_accuracy": acc, "pos": n_pos, "neg": n_neg}
            print(f"[phase4x] {name}: task_accuracy={acc:.3f}  answers={dist}  "
                  f"labeled pos={n_pos} neg={n_neg}")

    (config.RESULTS_DIR / "phase4x_generation.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[phase4x] wrote {out_path}")
    print("[phase4x] TASK ACCURACY per condition is the load-bearing-CoT control — if the illegible "
          "condition is much less accurate, reduced detectability may reflect degraded REASONING "
          "rather than reduced monitorability. Report both.")


if __name__ == "__main__":
    main()
