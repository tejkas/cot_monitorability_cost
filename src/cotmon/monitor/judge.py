"""Faithfulness gate (Phase 3) — the check we derived interactively.

A STRONG, INDEPENDENT judge (config.FAITHFULNESS_JUDGE_MODEL = Qwen3-14B) tries to recover
"was this trace swayed toward the hint?" from a tier's text. The reference is T0; the diagnostic
is the T0-vs-T1 recovery GAP: T1 is legible, so a drop there is the rewriter DELETING sway-signal,
not illegibility. It must be strong (near-ceiling on legible text) so a failure means the signal
is genuinely gone, and it's never a reported RQ1 result (so it stays independent of the monitor).

Implementation reuses the prompted TextMonitor with a stronger model; the judge is run as its own
step (one model in memory at a time — see scripts/05).
"""
from typing import List, Sequence

from .. import config
from .text_monitor import TextMonitor


class FaithfulnessJudge:
    def __init__(self, model: str = config.FAITHFULNESS_JUDGE_MODEL, **kw):
        self.reader = TextMonitor(model=model, **kw)

    def recover(self, texts: List[str], hint_targets: List[str], labels: Sequence[int]) -> float:
        """Recall on the TRUE positives: of genuinely-swayed traces, the fraction the judge still
        recovers as 'swayed' from `texts`. Compare across tiers vs T0 to get the recovery gap."""
        preds = self.reader.predict(texts, hint_targets)
        hits = [p for p, y in zip(preds, labels) if int(y) == 1]
        return sum(hits) / len(hits) if hits else float("nan")


def recovery_gap(recovery_t0: float, recovery_t1: float) -> float:
    """T0 - T1 recovery. Above config.FAITH_MAX_GAP => the rewriter is deleting sway-signal."""
    return recovery_t0 - recovery_t1
