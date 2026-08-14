#!/usr/bin/env python
"""Figures for the writeup. CPU only, reads the result JSONs.

  fig1_detectors.png  — AUROC per detector per condition, with question-clustered 95% CIs.
                        The story: the probe sits on top of the prompt-only floor.
  fig2_answers.png    — where each condition's answers land (gold/hint/other/none).
                        The story: the illegible model mostly stops solving the task, and its
                        "deference" positives are heavily diluted by near-random wrong answers
                        (Zolkowski's conditional-obfuscation point).

  python scripts/15_figures.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402

ORDER = ["prompt_only", "probe", "tfidf", "monitor_weak", "monitor_strong"]
NICE = {"prompt_only": "prompt-only\n(no reasoning)", "probe": "activation\nprobe",
        "tfidf": "TF-IDF", "monitor_weak": "monitor 3B", "monitor_strong": "monitor 14B"}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    res_path = config.RESULTS_DIR / "phase4x_analysis_stripped.json"
    if not res_path.exists():
        res_path = config.RESULTS_DIR / "phase4x_analysis.json"
    res = json.loads(res_path.read_text())
    gen = json.loads((config.RESULTS_DIR / "phase4x_generation.json").read_text())
    conds = [c for c in ("legible", "illegible") if c in res]

    # ---------------- fig 1: detectors ----------------
    fig, axes = plt.subplots(1, len(conds), figsize=(11, 4.6), sharey=True)
    for ax, cond in zip(np.atleast_1d(axes), conds):
        det = res[cond]["detectors"]
        names = [n for n in ORDER if n in det]
        vals = [det[n]["auroc"] for n in names]
        los = [det[n]["auroc"] - det[n]["ci"][0] for n in names]
        his = [det[n]["ci"][1] - det[n]["auroc"] for n in names]
        colors = ["#999999" if n == "prompt_only" else
                  "#2b6cb0" if n == "probe" else "#cbd5e0" for n in names]
        x = np.arange(len(names))
        ax.bar(x, vals, yerr=[los, his], capsize=4, color=colors, edgecolor="black", linewidth=0.6)
        # the prompt-only floor: anything at or below this line read nothing useful from the CoT
        floor = det.get("prompt_only", {}).get("auroc")
        if floor:
            ax.axhline(floor, ls="--", lw=1.2, color="#c53030")
            ax.text(len(names) - 0.4, floor + 0.008, "prompt-only floor",
                    ha="right", fontsize=8, color="#c53030")
        ax.axhline(0.5, ls=":", lw=1, color="black")
        ax.set_xticks(x)
        ax.set_xticklabels([NICE[n] for n in names], fontsize=8)
        acc = res[cond].get("task_accuracy")
        ax.set_title(f"{cond}  (task accuracy {acc:.2f})" if acc else cond, fontsize=11)
        ax.set_ylim(0.45, 1.0)
    np.atleast_1d(axes)[0].set_ylabel("AUROC")
    fig.suptitle("Detecting hint-deference: the probe barely clears a probe that never sees the reasoning",
                 fontsize=12)
    fig.tight_layout()
    p1 = config.RESULTS_DIR / "fig1_detectors.png"
    fig.savefig(p1, dpi=160)
    print(f"wrote {p1}")

    # ---------------- fig 2: where the answers land ----------------
    fig, ax = plt.subplots(figsize=(8, 3.4))
    keys = ["gold", "hint", "other", "none"]
    labels = {"gold": "correct", "hint": "the hint (deferred)",
              "other": "some other option", "none": "no parseable answer"}
    colors = {"gold": "#2f855a", "hint": "#c53030", "other": "#d69e2e", "none": "#a0aec0"}
    for i, cond in enumerate(conds):
        d = gen[cond]["answers"]
        total = sum(d.values())
        left = 0.0
        for k in keys:
            w = d[k] / total
            ax.barh(i, w, left=left, color=colors[k], edgecolor="white",
                    label=labels[k] if i == 0 else None)
            if w > 0.05:
                ax.text(left + w / 2, i, f"{d[k]}", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
            left += w
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([f"{c}\n(acc {gen[c]['task_accuracy']:.2f})" for c in conds], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of generated samples")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    ax.set_title("The illegible model mostly stops solving the task", fontsize=12)
    fig.tight_layout()
    p2 = config.RESULTS_DIR / "fig2_answers.png"
    fig.savefig(p2, dpi=160, bbox_inches="tight")
    print(f"wrote {p2}")

    # the conditional-obfuscation number, printed for the writeup
    print("\nDeference specificity — of WRONG answers, the share landing on the hint:")
    for c in conds:
        d = gen[c]["answers"]
        wrong = d["hint"] + d["other"]
        print(f"  {c:10s} {d['hint']}/{wrong} = {d['hint'] / wrong:.3f}"
              f"   (chance with ~9 wrong options ≈ 0.111)")


if __name__ == "__main__":
    main()
