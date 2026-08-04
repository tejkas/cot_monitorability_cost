#!/usr/bin/env python
"""Phase 1 — generate hint-usage CoT traces and label them (DESIGN.md §4).

Stage A: sample k unhinted generations per question -> keep "capable" questions
         (baseline confidently correct).
Stage B: for each capable question x hint family, point a hint at a wrong option,
         sample k hinted generations, and label each trace
         (a1==h -> positive, a1==gold -> negative, else excluded).

Outputs (JSONL, under data/traces/):
  phase1_traces.jsonl     one row per labeled trace (the training/eval data)
  phase1_questions.jsonl  per-question stats (P0/P1, lift) — for tuning k & threshold

Run on the GPU box:
  python scripts/01_generate_traces.py --n-questions 300 --k 8
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402
from cotmon.data import generate as gen  # noqa: E402
from cotmon.data import labeling as lab  # noqa: E402
from cotmon.data.hints import make_hint  # noqa: E402


def load_questions(n: int, seed: int):
    """Load and shuffle n MMLU-Pro questions.

    NOTE: verify field names on the box — MMLU-Pro exposes `question`, `options`
    (list), `answer` (letter), `answer_index`, `category`. Some dumps pad `options`
    with "N/A"; handle that here if it appears.
    """
    from datasets import load_dataset

    ds = load_dataset(config.PRIMARY_DATASET, split="test")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx[:n]:
        r = ds[i]
        # Some MMLU-Pro dumps pad `options` to 10 with "N/A"; drop trailing padding so
        # n_options, the offered letters, and hint targets all reflect the REAL options.
        # (Padding is always trailing and gold is a real option, so this can't shift gold.)
        opts = list(r["options"])
        while opts and str(opts[-1]).strip().upper() in {"N/A", ""}:
            opts.pop()
        out.append({
            "question_id": r.get("question_id", i),
            "question": r["question"],
            "options": opts,
            "gold": r["answer"],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=config.N_QUESTIONS)
    ap.add_argument("--k", type=int, default=config.K_SAMPLES)
    ap.add_argument("--model", default=config.BASE_MODEL)
    ap.add_argument("--out-dir", default=str(config.DATA_DIR / "traces"))
    args = ap.parse_args()

    min_valid = max(1, args.k // 2)  # drop questions with too many unparseable samples
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qs = load_questions(args.n_questions, config.SEED)
    G = gen.Generator(model=args.model)

    # --- Stage A: unhinted -> capable filter ---
    user_turns = [gen.build_user_turn(q["question"], q["options"]) for q in qs]
    n_opts = [len(q["options"]) for q in qs]
    unhinted = G.sample(user_turns, args.k, n_opts)

    capable, q_summ = [], []
    for q, samples, n_opt in zip(qs, unhinted, n_opts):
        p0, n_valid = lab.distribution(samples, n_opt)
        ok = lab.is_capable(p0, q["gold"], n_valid, min_valid)
        q_summ.append({"question_id": q["question_id"], "gold": q["gold"],
                       "n_valid": n_valid, "p0": p0, "capable": ok})
        if ok:
            a0_star, _ = lab.mode_and_peak(p0)
            capable.append({**q, "p0": p0, "a0_star": a0_star, "n_opt": n_opt})
    print(f"[phase1] capable questions: {len(capable)}/{len(qs)}")

    # --- Stage B: hinted per family -> label ---
    rng = random.Random(config.SEED)
    traces_path = out_dir / "phase1_traces.jsonl"
    n_pos = n_neg = 0
    fam_lift = {}  # (question_id, family) -> {h, p1, lift}

    with traces_path.open("w") as f:
        for family in config.HINT_FAMILIES:
            jobs = []
            for q in capable:
                letters = [chr(ord("A") + i) for i in range(q["n_opt"])]
                # Redraw h INDEPENDENTLY per family (deliberate — do NOT hoist to per-question).
                # Independent targets spread hints across answer choices and decorrelate
                # "hint family" from "which option was targeted", so the cross-family
                # comparison isn't confounded by any option-specific effect.
                h = rng.choice([L for L in letters if L != q["gold"]])  # a WRONG option
                turn = gen.build_user_turn(q["question"], q["options"], make_hint(family, h))
                jobs.append((q, h, turn))

            hinted = G.sample([j[2] for j in jobs], args.k, [j[0]["n_opt"] for j in jobs])
            for (q, h, _), samples in zip(jobs, hinted):
                p1, _ = lab.distribution(samples, q["n_opt"])
                fam_lift[(q["question_id"], family)] = {
                    "hint_target": h, "p1": p1, "lift": p1.get(h, 0.0) - q["p0"].get(h, 0.0)}
                for rec in lab.label_traces(q["question_id"], family, h, q["a0_star"],
                                            q["p0"], p1, samples):
                    f.write(json.dumps(rec) + "\n")
                    if rec["label"] == 1:
                        n_pos += 1
                    else:
                        n_neg += 1

    # attach per-family lift to the question summary and write it out
    for row in q_summ:
        row["hinted"] = {fam: fam_lift.get((row["question_id"], fam))
                         for fam in config.HINT_FAMILIES
                         if (row["question_id"], fam) in fam_lift}
    with (out_dir / "phase1_questions.jsonl").open("w") as f:
        for row in q_summ:
            f.write(json.dumps(row) + "\n")

    print(f"[phase1] wrote {traces_path}  (positives={n_pos}, negatives={n_neg})")
    print(f"[phase1] wrote {out_dir / 'phase1_questions.jsonl'}")


if __name__ == "__main__":
    main()
