#!/usr/bin/env python
"""Phase 4c — the LoRA'd model GENERATES genuine T3-style CoT on HELD-OUT hinted questions.

Loads base Qwen3-8B + the T3 LoRA adapter and regenerates the TEST-split (question, hint) items
with the fine-tuned model, labelling behaviorally (answer == hint -> swayed). This is the genuine
article: a model that ACTUALLY reasons in near-illegible style, not a re-encoded rewrite. The
sway is not trained -> whatever the probe recovers here is real, not adversarial obfuscation.

Reuses Phase-1 generation (with our fixed answer extraction) + labeling. Output:
  data/traces/genuine_traces.jsonl   (one labeled genuine trace per valid sample)

  python scripts/08_generate_genuine.py                 # full test split
  python scripts/08_generate_genuine.py --limit 20      # smoke
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402
from cotmon.data import generate as gen  # noqa: E402
from cotmon.data import labeling as lab  # noqa: E402
from cotmon.data.hints import make_hint  # noqa: E402
from cotmon.probe.sweep import question_split  # noqa: E402


def load_mmlu_index() -> dict:
    """question_id -> {question, options, gold} (trailing 'N/A' padding stripped, as in Phase 1)."""
    from datasets import load_dataset

    ds = load_dataset(config.PRIMARY_DATASET, split="test")
    idx = {}
    for i, r in enumerate(ds):
        opts = list(r["options"])
        while opts and str(opts[-1]).strip().upper() in {"N/A", ""}:
            opts.pop()
        idx[r.get("question_id", i)] = {"question": r["question"], "options": opts,
                                        "gold": r["answer"]}
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=config.K_SAMPLES)
    ap.add_argument("--lora", default=str(config.RESULTS_DIR / config.LORA_OUT))
    ap.add_argument("--in", dest="in_path",
                    default=str(config.DATA_DIR / "traces" / "phase2_tiers.jsonl"))
    ap.add_argument("--out", default=str(config.DATA_DIR / "traces" / "genuine_traces.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="cap unique (q,hint) items for a smoke run")
    ap.add_argument("--temp", type=float, default=0.3,
                    help="generation temperature (low keeps style; >0 also helps break loops)")
    ap.add_argument("--rep-penalty", type=float, default=1.3,
                    help="repetition penalty; the symbolic T3 style loops without it (Phase 2 used 1.3)")
    args = ap.parse_args()

    # Held-out items only: same question-level split the probe uses, so the LoRA never saw them.
    rows = [json.loads(l) for l in open(args.in_path)]
    qids = np.array([r["question_id"] for r in rows])
    _, te = question_split(qids, config.PROBE_TEST_FRACTION, config.PROBE_SPLIT_SEED)
    te_rows = set(te.tolist())

    seen, items = set(), []
    for i, r in enumerate(rows):
        if i not in te_rows:
            continue
        key = (r["question_id"], r["hint_family"], r["hint_target"])
        if key in seen:
            continue
        seen.add(key)
        items.append(r)
    if args.limit:
        items = items[:args.limit]
    print(f"[phase4c] {len(items)} unique held-out (question, hint) items -> {args.k} samples each")

    mmlu = load_mmlu_index()
    jobs, turns, nopts = [], [], []
    for r in items:
        q = mmlu.get(r["question_id"])
        if q is None:
            continue
        turns.append(gen.build_user_turn(q["question"], q["options"],
                                         make_hint(r["hint_family"], r["hint_target"])))
        nopts.append(len(q["options"]))
        jobs.append((r, q))

    G = gen.Generator(model=config.BASE_MODEL, lora=args.lora)
    print(f"[phase4c] generating with LoRA at temp={args.temp}, rep_penalty={args.rep_penalty}")
    samples = G.sample(turns, args.k, nopts, temperature=args.temp,
                       repetition_penalty=args.rep_penalty)

    # Loop diagnostic: a looping symbolic trace runs to the token cap -> finish_reason "length",
    # no </think>, no parseable answer -> silently dropped. Surface it instead of hiding it.
    flat = [s for samps in samples for s in samps]
    n_trunc = sum(1 for s in flat if s.finish_reason == "length")
    n_noans = sum(1 for s in flat if s.answer is None)
    cot_lens = sorted(len(s.cot) for s in flat)
    print(f"[phase4c] {len(flat)} samples: truncated(loops)={n_trunc}, no-answer={n_noans}, "
          f"cot chars med={cot_lens[len(cot_lens) // 2] if cot_lens else 0}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_pos = n_neg = 0
    with out_path.open("w") as f:
        for (r, q), samps in zip(jobs, samples):
            for rec in lab.label_traces(r["question_id"], r["hint_family"], r["hint_target"],
                                        q["gold"], {}, {}, samps):  # p0/p1 empty -> lift unused here
                f.write(json.dumps(rec) + "\n")
                n_pos += int(rec["label"] == 1)
                n_neg += int(rec["label"] == 0)
    print(f"[phase4c] wrote {out_path}  (pos={n_pos}, neg={n_neg}, total={n_pos + n_neg})")


if __name__ == "__main__":
    main()
