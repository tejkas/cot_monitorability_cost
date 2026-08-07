"""Rewrite T0 CoTs into legibility tiers via a SEPARATE instruct LLM (Phase 2).

Distinct from data/generate.py (base model, thinking ON): here we run a smaller INSTRUCT
model, thinking OFF, one low-temperature completion per (trace, tier). vLLM is imported
lazily so the module imports on a laptop for syntax checks.
"""
import re
from typing import Dict, List

from transformers import AutoTokenizer

from .. import config
from .prompts import SYSTEM, TIER_PROMPTS

# Tiers we actually generate (T0 is the verbatim original, not rewritten).
REWRITE_TIERS = [t for t in config.TIER_NAMES if t != "T0_verbatim"]

# Strip a leading "Here is the rewritten trace:" / "Sure! ..." if the model adds one.
_PREAMBLE = re.compile(r"^\s*(here(?:'s| is)[^\n:]*:|rewritten[^\n:]*:|sure[!,.][^\n]*)\s*", re.I)


def strip_preamble(text: str) -> str:
    return _PREAMBLE.sub("", text.strip(), count=1).strip()


class Rewriter:
    """Thin wrapper over vLLM offline generation with a non-thinking instruct model."""

    def __init__(self, model: str = config.REWRITER_MODEL,
                 max_model_len: int = config.MAX_MODEL_LEN, gpu_mem_util: float = 0.90):
        from vllm import LLM  # lazy import

        self.tok = AutoTokenizer.from_pretrained(model)
        self.llm = LLM(model=model, dtype="float16",
                       gpu_memory_utilization=gpu_mem_util, max_model_len=max_model_len)

    def _template(self, cot: str, tier: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": TIER_PROMPTS[tier].format(cot=cot)}],
            tokenize=False, add_generation_prompt=True,
        )

    def rewrite(self, cots: List[str]) -> List[Dict[str, str]]:
        """For each input CoT return {tier_name: text} for all 4 tiers (T0 = verbatim).

        All (trace, tier) prompts are flattened into one batched vLLM call for throughput,
        then re-assembled back to per-trace dicts.
        """
        from vllm import SamplingParams

        # Per-tier sampling: prose (T1) needs NO repetition penalty; symbolic tiers need one to
        # avoid loops. vLLM accepts a list of SamplingParams aligned 1:1 with the prompts.
        def _sp(tier: str) -> "SamplingParams":
            return SamplingParams(
                n=1, temperature=config.REWRITE_TEMPERATURE, top_p=config.REWRITE_TOP_P,
                max_tokens=config.REWRITE_MAX_TOKENS, seed=config.SEED,
                repetition_penalty=config.REWRITE_REPETITION_PENALTY.get(tier, 1.0),
            )
        sp_by_tier = {tier: _sp(tier) for tier in REWRITE_TIERS}

        prompts: List[str] = []
        params: list = []
        index: List[tuple] = []  # (trace_i, tier)
        for i, cot in enumerate(cots):
            for tier in REWRITE_TIERS:
                prompts.append(self._template(cot, tier))
                params.append(sp_by_tier[tier])
                index.append((i, tier))

        outs = self.llm.generate(prompts, params)

        result: List[Dict[str, str]] = [{"T0_verbatim": c} for c in cots]
        for (i, tier), out in zip(index, outs):
            result[i][tier] = strip_preamble(out.outputs[0].text)
        return result
