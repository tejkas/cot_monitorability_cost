"""Offline bulk generation of CoT traces (Phase 1), via vLLM.

This is OFFLINE batched sampling — we only need text out. It is distinct from:
  - Phase 3 HF activation extraction (white-box, `output_hidden_states`), and
  - the Phase 5 serving-cost harness (continuous batching + activation extraction).
Using vLLM here is purely for throughput on bulk sampling.

vLLM is imported lazily inside `Generator` so this module imports fine on a laptop
(for syntax checks) without vLLM/CUDA installed.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

from transformers import AutoTokenizer

from .. import config

LETTERS = [chr(ord("A") + i) for i in range(26)]


def build_user_turn(question: str, options: List[str], hint_text: Optional[str] = None) -> str:
    """The MCQ user turn. The hint (if any) is appended before the answer instruction."""
    lines = [f"Question: {question}", "Options:"]
    for i, opt in enumerate(options):
        lines.append(f"({LETTERS[i]}) {opt}")
    if hint_text:
        lines.append(hint_text)
    lines.append("Answer with the letter of the correct option.")
    return "\n".join(lines)


# Answer-extraction patterns, tried in order (most specific first).
_ANSWER_PATTERNS = [
    re.compile(r"answer\s*is\s*:?\s*\(?([A-Z])\)?", re.I),
    re.compile(r"\*\*\s*\(?([A-Z])\)?\s*\*\*"),
    re.compile(r"\b([A-Z])\)"),
    re.compile(r"^\s*\(?([A-Z])\)?\s*$", re.M),
]


def split_cot_answer(raw: str) -> tuple[str, str]:
    """Split a raw completion into (cot_text, answer_text).

    Handles the Qwen3-4B-Thinking quirk where only the closing </think> is emitted.
    If </think> never appears the generation was likely truncated: treat it all as
    'thinking' with no answer.
    """
    if "</think>" in raw:
        pre, post = raw.split("</think>", 1)
        return pre.replace("<think>", "").strip(), post.strip()
    return raw.replace("<think>", "").strip(), ""


def parse_answer(answer_text: str, n_options: int) -> Optional[str]:
    """Extract the chosen option letter from the post-</think> text, or None."""
    valid = set(LETTERS[:n_options])
    for pat in _ANSWER_PATTERNS:
        for m in pat.finditer(answer_text):
            if m.group(1).upper() in valid:
                return m.group(1).upper()
    return None


@dataclass
class Sample:
    answer: Optional[str]  # parsed option letter, or None if unparseable/truncated
    cot: str               # the <think> content — what monitors/probes read later
    raw: str               # full raw completion (kept for debugging)


class Generator:
    """Thin wrapper over vLLM offline generation with Qwen3 thinking enabled."""

    def __init__(self, model: str = config.BASE_MODEL, max_model_len: int = config.MAX_MODEL_LEN,
                 gpu_mem_util: float = 0.90):
        from vllm import LLM  # lazy import

        self.tok = AutoTokenizer.from_pretrained(model)
        self.llm = LLM(model=model, dtype="float16",
                       gpu_memory_utilization=gpu_mem_util, max_model_len=max_model_len)

    def _template(self, user_turn: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user_turn}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )

    def sample(self, user_turns: List[str], k: int,
               n_options_list: List[int]) -> List[List[Sample]]:
        """For each user turn, return k parsed Samples. Batched across all turns."""
        from vllm import SamplingParams

        sp = SamplingParams(
            n=k, temperature=config.GEN_TEMPERATURE, top_p=config.GEN_TOP_P,
            top_k=config.GEN_TOP_K, max_tokens=config.GEN_MAX_NEW_TOKENS, seed=config.SEED,
        )
        prompts = [self._template(u) for u in user_turns]
        outs = self.llm.generate(prompts, sp)

        results: List[List[Sample]] = []
        for out, n_opt in zip(outs, n_options_list):
            samples = []
            for comp in out.outputs:
                cot, ans_text = split_cot_answer(comp.text)
                samples.append(Sample(answer=parse_answer(ans_text, n_opt), cot=cot, raw=comp.text))
            results.append(samples)
        return results
