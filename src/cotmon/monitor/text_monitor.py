"""Prompted text monitor (Phase 3) — the black-box 'text' reader in the RQ1 divergence.

A SEPARATE small model (config.TEXT_MONITOR_MODEL = SmolLM3-3B — different family from the
base, the rewriter, and the judge, so it can't get an unfair edge) reads a tier's TEXT and
predicts whether the trace was swayed toward its hint. Its accuracy-vs-tier decline is the
'text' curve we compare the probe against. Batched via vLLM; guided yes/no for a clean readout.

--- interface fixed here; implementation built in Phase 3 ---
"""
from typing import List

from .. import config


class TextMonitor:
    def __init__(self, model: str = config.TEXT_MONITOR_MODEL):
        self.model_name = model
        raise NotImplementedError("build in Phase 3")

    def predict(self, texts: List[str], hint_targets: List[str]) -> List[bool]:
        """Per text: does the monitor judge the reasoning was swayed toward its hint?"""
        raise NotImplementedError("build in Phase 3")
