#!/usr/bin/env python
"""Phase 4 (expanded) — the single analysis: probe vs text detectors, legible vs illegible.

One table. Every column exists for a stated reason:

  probe            activation probe, trained on Set B and evaluated on Set C, WITHIN condition
                   (the standard probing setup: same distribution, disjoint questions)
  prompt_only      floor: what is knowable from the question+hint BEFORE any reasoning exists.
                   Nothing below this line is evidence that a detector read the reasoning.
  tfidf            floor: what is knowable from surface word statistics alone (McKenzie found
                   probes DO fire on surface keywords, so this must be measured)
  monitor_weak     prompted LLM monitor (the deployment-realistic, cheaper-than-the-model case)
  monitor_strong   prompted LLM monitor, large — the fair upper bound for text-based detection
  task_accuracy    the load-bearing-CoT control: if the illegible condition is much less accurate,
                   reduced detectability may mean degraded REASONING, not reduced monitorability

Metrics: AUROC (probe-literature primary) + TPR@1%FPR (deployment operating point, used by both
literatures) + 95% CIs bootstrapped over QUESTIONS, since k samples per question are correlated.

Activations are teacher-forced over `prompt + CoT` and pooled over CoT positions only — matching
the model's generation-time state, with the question and hint in context.

  python scripts/14_analyze_conditions.py
  python scripts/14_analyze_conditions.py --strip-conclusion --strong-monitor
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402
from cotmon.data.generate import build_user_turn, strip_conclusion  # noqa: E402
from cotmon.data.hints import make_hint  # noqa: E402
from cotmon.monitor import surface as surf  # noqa: E402
from cotmon.monitor import text_monitor as tm  # noqa: E402
from cotmon.probe import probe as P  # noqa: E402
from cotmon.probe.extract import ActivationExtractor  # noqa: E402

RNG = np.random.RandomState(config.SEED)


def boot_ci(y, s, q, n_boot=2000, fn=None):
    """95% CI resampling QUESTIONS (clusters), not traces."""
    fn = fn or P.auroc
    qs = np.asarray(q)
    uniq = np.unique(qs)
    idx = {u: np.where(qs == u)[0] for u in uniq}
    vals = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[u] for u in RNG.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[rows])) < 2:
            continue
        vals.append(fn(y[rows], s[rows]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default=str(config.DATA_DIR / "traces" / "conditions.jsonl"))
    ap.add_argument("--strip-conclusion", action="store_true",
                    help="remove each CoT's own answer statement from EVERY detector's view")
    ap.add_argument("--strong-monitor", nargs="?", const=config.FAITHFULNESS_JUDGE_MODEL, default=None)
    ap.add_argument("--no-monitor", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.in_path)]
    conditions = sorted({r["condition"] for r in rows})
    gen_stats = {}
    gpath = config.RESULTS_DIR / "phase4x_generation.json"
    if gpath.exists():
        gen_stats = json.loads(gpath.read_text())

    from datasets import load_dataset
    ds = load_dataset(config.PRIMARY_DATASET, split="test")
    qinfo = {}
    for i, r in enumerate(ds):
        opts = list(r["options"])
        while opts and str(opts[-1]).strip().upper() in {"N/A", ""}:
            opts.pop()
        qinfo[r.get("question_id", i)] = {"question": r["question"], "options": opts}

    # ---------------- PHASE A: all activations first, with ONE HF model loaded ----------------
    # The extractor holds ~16GB of weights. vLLM then asks for 90% of the card, so the extractor
    # MUST be freed before any monitor loads (this exact overlap OOM'd an earlier script).
    extractor = ActivationExtractor()
    per_cond = {}

    for cond in conditions:
        sub = [r for r in rows if r["condition"] == cond]
        y = np.array([r["label"] for r in sub])
        q = np.array([r["question_id"] for r in sub])
        sets = np.array([r["set"] for r in sub])
        cots = [strip_conclusion(r["cot_text"]) if args.strip_conclusion else r["cot_text"]
                for r in sub]
        prompts = [extractor.chat_prompt(build_user_turn(
            qinfo[r["question_id"]]["question"], qinfo[r["question_id"]]["options"],
            make_hint(r["hint_family"], r["hint_target"]))) for r in sub]

        tr = np.where(sets == "B")[0]     # probe training questions
        te = np.where(sets == "C")[0]     # evaluation questions (disjoint by construction)
        y_tr, y_te, q_te = y[tr], y[te], q[te]
        print(f"\n=== {cond}: n={len(sub)}  train(B) n={len(tr)} pos={int(y_tr.sum())}  "
              f"eval(C) n={len(te)} pos={int(y_te.sum())} "
              f"questions_with_pos={len(np.unique(q_te[y_te == 1]))} ===", flush=True)

        # --- activations: prompt+CoT teacher-forced, pooled over CoT positions ---
        # n in the filename: a small smoke run must never silently poison a full run's cache
        tag = f"{cond}_n{len(sub)}{'_stripped' if args.strip_conclusion else ''}"
        cache = config.ACTIVATIONS_DIR / f"acts_{tag}.npz"
        if cache.exists():
            z = np.load(cache)
            acts = {p: z[p] for p in config.PROBE_POOLINGS}
            print(f"[phase4x] loaded {cache}")
        else:
            acts = extractor.extract(cots, prompts=prompts)
            config.ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **{p: acts[p].astype(config.ACT_DTYPE) for p in acts})
            print(f"[phase4x] wrote {cache}")

        # --- prompt-only floor activations: what is knowable from question+hint alone ---
        pcache = config.ACTIVATIONS_DIR / f"acts_{cond}_n{len(sub)}_promptonly.npz"
        if pcache.exists():
            pacts = {p: np.load(pcache)[p] for p in config.PROBE_POOLINGS}
            print(f"[phase4x] loaded {pcache}")
        else:
            pacts = extractor.extract(prompts)   # no completion -> question+hint only
            np.savez(pcache, **{p: pacts[p].astype(config.ACT_DTYPE) for p in pacts})
            print(f"[phase4x] wrote {pcache}")

        per_cond[cond] = {"sub": sub, "y": y, "q": q, "cots": cots, "tr": tr, "te": te,
                          "acts": acts, "pacts": pacts}

    # free the HF model before any vLLM engine starts
    del extractor
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    print("\n[phase4x] extractor released; GPU free for the text monitors\n", flush=True)

    # ---------------- PHASE B: probes + surface baseline (CPU only) ----------------
    results = {}
    for cond in conditions:
        d = per_cond[cond]
        sub, y, q, cots = d["sub"], d["y"], d["q"], d["cots"]
        tr, te, acts, pacts = d["tr"], d["te"], d["acts"], d["pacts"]
        y_tr, y_te, q_te = y[tr], y[te], q[te]

        # --- probe: sweep cells INSIDE the training set, then report that cell on eval ---
        # The inner split is by QUESTION, not by row: k traces share a question, so a row split
        # would leak siblings between selection-train and selection-val and bias the cell choice.
        q_tr = q[tr]
        uq = np.unique(q_tr)
        inner_val_q = set(uq[: max(1, len(uq) // 3)].tolist())
        in_val = np.array([qq in inner_val_q for qq in q_tr])
        best = (-1.0, None, None)
        if in_val.any() and (~in_val).any() and len(np.unique(y_tr[in_val])) > 1 \
                and len(np.unique(y_tr[~in_val])) > 1:
            for pool in config.PROBE_POOLINGS:
                for j, layer in enumerate(config.PROBE_LAYERS):
                    m = P.train_eval(acts[pool][tr[~in_val], j, :], y_tr[~in_val],
                                     acts[pool][tr[in_val], j, :], y_tr[in_val], config.PROBE_KIND)
                    if m["auroc"] == m["auroc"] and m["auroc"] > best[0]:
                        best = (m["auroc"], pool, layer)
        _, pool, layer = best
        if pool is None:
            # too little/too degenerate training data to select a cell (e.g. a smoke run):
            # fall back to the configured default rather than crashing downstream.
            pool, layer = config.PROBE_POOLINGS[0], config.PROBE_LAYERS[len(config.PROBE_LAYERS) // 2]
            print(f"[phase4x] WARNING: cell selection failed (train too small/degenerate); "
                  f"falling back to {pool}/L{layer}", flush=True)
        j = config.PROBE_LAYERS.index(layer)
        pr = P._PROBES[config.PROBE_KIND]().fit(np.asarray(acts[pool][tr, j, :], np.float32), y_tr)
        s_probe = pr.decision_scores(np.asarray(acts[pool][te, j, :], np.float32))

        # prompt-only floor: identical probe recipe, activations of the PROMPT alone
        ppr = P._PROBES[config.PROBE_KIND]().fit(np.asarray(pacts[pool][tr, j, :], np.float32), y_tr)
        s_prompt = ppr.decision_scores(np.asarray(pacts[pool][te, j, :], np.float32))

        s_tfidf = surf.TfidfSurfaceBaseline().fit([cots[i] for i in tr], y_tr).scores(
            [cots[i] for i in te])

        d["scores"] = {"probe": s_probe, "prompt_only": s_prompt, "tfidf": s_tfidf}
        results[cond] = {"cell": {"pooling": pool, "layer": layer,
                                  "depth_fraction": round(layer / 36, 2)},
                         "n_eval": len(te), "pos_eval": int(y_te.sum()),
                         "questions_with_pos": int(len(np.unique(q_te[y_te == 1]))),
                         "task_accuracy": gen_stats.get(cond, {}).get("task_accuracy")}

    # ---------------- PHASE C: text monitors — load each model ONCE, score every condition ----
    if not args.no_monitor:
        for tag, mdl in [("weak", config.TEXT_MONITOR_MODEL), ("strong", args.strong_monitor)]:
            if mdl is None:
                continue
            print(f"[phase4x] monitor '{tag}': {mdl}", flush=True)
            M = tm.TextMonitor(model=mdl, strip_conclusion=False)  # cots already stripped above
            for cond in conditions:
                d = per_cond[cond]
                te, sub, cots = d["te"], d["sub"], d["cots"]
                d["scores"][f"monitor_{tag}"] = np.array(M.predict_scores(
                    [cots[i] for i in te], [sub[i]["hint_target"] for i in te]))
            del M
            gc.collect()
            torch.cuda.empty_cache()

    # ---------------- assemble ----------------
    for cond in conditions:
        d = per_cond[cond]
        y_te, q_te = d["y"][d["te"]], d["q"][d["te"]]
        det = {}
        for name, s in d["scores"].items():
            lo, hi = boot_ci(y_te, s, q_te)
            det[name] = {"auroc": P.auroc(y_te, s), "ci": [lo, hi],
                         "tpr@1fpr": P.tpr_at_fpr(y_te, s, 0.01)}
        results[cond]["detectors"] = det

    # ---- report ----
    print("\n" + "=" * 96)
    print(f"{'condition':11s}{'detector':16s}{'AUROC':>8s}{'95% CI':>18s}{'TPR@1%FPR':>11s}"
          f"{'task_acc':>10s}{'eval_q_pos':>11s}")
    print("-" * 96)
    for cond in conditions:
        r = results[cond]
        ta = r["task_accuracy"]
        ta_s = f"{ta:.3f}" if ta is not None else "-"
        for name, d in r["detectors"].items():
            ci_s = f"[{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"
            print(f"{cond:11s}{name:16s}{d['auroc']:>8.3f}{ci_s:>18s}"
                  f"{d['tpr@1fpr']:>11.3f}{ta_s:>10s}{r['questions_with_pos']:>11d}")
        print("-" * 96)
    print("READ: the probe must beat prompt_only AND tfidf to be doing anything semantic, and must")
    print("beat monitor_strong to support 'probes see what text monitoring cannot'.")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / f"phase4x_analysis{'_stripped' if args.strip_conclusion else ''}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[phase4x] wrote {out}")


if __name__ == "__main__":
    main()
