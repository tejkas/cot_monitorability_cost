"""Faithfulness gate (Phase 3) — the check we derived interactively.

A STRONG, INDEPENDENT judge (config.FAITHFULNESS_JUDGE_MODEL = Qwen3-14B) tries to recover
"was this trace swayed toward the hint?" from a tier's text. The reference is T0; the
diagnostic is the T0-vs-T1 recovery GAP: T1 is legible, so a drop there is the rewriter
DELETING sway-signal, not illegibility. A decompression/round-trip variant probes T2/T3 more
directly. It must be strong (near-ceiling on legible text) so a failure means the signal is
gone, and never a reported RQ1 result (so it stays independent of the text monitor).

--- interface fixed here; judge model call built in Phase 3 ---
"""
from typing import List, Sequence

from .. import config


class FaithfulnessJudge:
    def __init__(self, model: str = config.FAITHFULNESS_JUDGE_MODEL):
        self.model_name = model
        raise NotImplementedError("build in Phase 3")

    def recover(self, texts: List[str], hint_targets: List[str], labels: Sequence[int]) -> float:
        """Fraction of the true-positive traces the judge still recovers as 'swayed' from `texts`."""
        raise NotImplementedError("build in Phase 3")


def recovery_gap(recovery_t0: float, recovery_t1: float) -> float:
    """T0 - T1 recovery. Above config.FAITH_MAX_GAP => the rewriter is deleting sway-signal."""
    return recovery_t0 - recovery_t1
