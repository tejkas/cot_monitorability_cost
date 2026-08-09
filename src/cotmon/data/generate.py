"""Offline bulk generation of CoT traces (Phase 1), via vLLM.

This is OFFLINE batched sampling — we only need text out. It is distinct from:
  - Phase 3 HF activation extraction (white-box, `output_hidden_states`), and
  - the Phase 5 serving-cost harness (continuous batching + activation extraction).
Using vLLM here is purely for throughput on bulk sampling.

vLLM is imported lazily inside `Generator` so this module imports fine on a laptop
(for syntax checks) without vLLM/CUDA installed.
"""
import math
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
    lines.append("End your reply with 'Final answer: (X)', where X is the letter of the correct option.")
    return "\n".join(lines)


# Answer-extraction patterns, tried in priority order (most specific first).
_ANSWER_PATTERNS = [
    re.compile(r"final\s*answer\s*[:\-]?\s*\(?([A-Z])\)?", re.I),  # the explicit anchor we now request
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
    """Extract the model's FINAL chosen letter from the post-</think> text, or None.

    Take the LAST valid match of the highest-priority pattern present. The model states its
    own answer LAST — after any echo of the hint (e.g. "the professor said (H), but the answer
    is (D)"). The old first-match logic grabbed the echoed hint and silently mislabeled correct
    answers as swayed — the Phase-1 positive-class contamination bug (found 2026-08-06).
    """
    valid = set(LETTERS[:n_options])
    for pat in _ANSWER_PATTERNS:
        hits = [m.group(1).upper() for m in pat.finditer(answer_text) if m.group(1).upper() in valid]
        if hits:
            return hits[-1]
    return None


def _letter_from_tok(s: str, valid: set) -> Optional[str]:
    """Reduce a decoded token (' D', '(D', 'D)', …) to one valid option letter, or None."""
    t = s.strip().strip("()[]{}:.,*# ").upper()
    return t if len(t) == 1 and t in valid else None


def answer_confidence(comp, valid: set) -> "tuple[Optional[str], Optional[float]]":
    """Read the committed answer + its confidence from a vLLM completion carrying logprobs.

    Locate the first valid-letter token AFTER the last 'Final answer' anchor (the model's
    commitment — distinct from any echoed 'the professor's answer is X'), and return
    (letter, margin) where margin = P(top letter) - P(runner-up letter) at that token. A small
    margin => the model was ~coin-flipping (flaky) and the label there is untrustworthy.
    Returns (None, None) if logprobs / anchor / letter can't be found.
    """
    lps = getattr(comp, "logprobs", None)
    toks = getattr(comp, "token_ids", None)
    if not lps or not toks:
        return None, None
    dec = []
    for i, tid in enumerate(toks):
        entry = lps[i] if i < len(lps) else None
        lp = entry.get(tid) if entry else None
        dec.append(lp.decoded_token if (lp and lp.decoded_token) else "")

    anchor_end = None
    for m in re.finditer(r"final\s*answer", "".join(dec), re.I):
        anchor_end = m.end()
    if anchor_end is None:
        return None, None

    cum = 0
    for i, s in enumerate(dec):
        start = cum
        cum += len(s)
        if start < anchor_end:
            continue
        letter = _letter_from_tok(s, valid)
        if letter is None:
            continue
        probs: dict = {}
        for tid, lp in (lps[i] or {}).items():
            ch = _letter_from_tok(lp.decoded_token or "", valid)
            if ch:
                probs[ch] = max(probs.get(ch, 0.0), math.exp(lp.logprob))
        ranked = sorted(probs.values(), reverse=True)
        top = ranked[0] if ranked else 0.0
        second = ranked[1] if len(ranked) > 1 else 0.0
        return letter, top - second
    return None, None


@dataclass
class Sample:
    answer: Optional[str]  # committed option letter (from the 'Final answer' token), or None
    cot: str               # the <think> content — what monitors/probes read later
    raw: str               # full raw completion (kept for debugging)
    finish_reason: Optional[str] = None  # vLLM: "length" == hit token cap (TRUNCATED); "stop" == completed
    answer_text: str = ""              # post-</think> text — stored so answers can be re-audited
    conf_margin: Optional[float] = None  # P(top letter) - P(runner-up) at the answer token
    flaky: bool = False                # True if conf_margin < ANSWER_CONF_MARGIN (model ~coin-flipped)


class Generator:
    """Thin wrapper over vLLM offline generation with Qwen3 thinking enabled."""

    def __init__(self, model: str = config.BASE_MODEL, max_model_len: int = config.MAX_MODEL_LEN,
                 gpu_mem_util: float = 0.90, lora: Optional[str] = None):
        from vllm import LLM  # lazy import

        self.tok = AutoTokenizer.from_pretrained(model)
        self.lora = lora  # path to a LoRA adapter (Phase 4 genuine generation), or None for base
        kw = dict(enable_lora=True, max_lora_rank=config.LORA_R) if lora else {}
        self.llm = LLM(model=model, dtype="float16",
                       gpu_memory_utilization=gpu_mem_util, max_model_len=max_model_len, **kw)

    def _template(self, user_turn: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user_turn}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )

    def sample(self, user_turns: List[str], k: int, n_options_list: List[int],
               temperature: Optional[float] = None) -> List[List[Sample]]:
        """For each user turn, return k parsed Samples. Batched across all turns.

        temperature overrides config.GEN_TEMPERATURE — Phase 4 genuine generation uses a LOWER
        temp so the LoRA's compressed T3 style survives sampling (at 0.6 the base verbose prior
        wins; at greedy the LoRA style is perfect).
        """
        from vllm import SamplingParams

        sp = SamplingParams(
            n=k, temperature=config.GEN_TEMPERATURE if temperature is None else temperature,
            top_p=config.GEN_TOP_P, top_k=config.GEN_TOP_K,
            max_tokens=config.GEN_MAX_NEW_TOKENS, seed=config.SEED,
            logprobs=config.ANSWER_LOGPROBS,  # top-k at each token -> read the answer letter's margin
        )
        prompts = [self._template(u) for u in user_turns]
        outs = self.llm.generate(prompts, sp)

        results: List[List[Sample]] = []
        for out, n_opt in zip(outs, n_options_list):
            valid = set(LETTERS[:n_opt])
            samples = []
            for comp in out.outputs:
                cot, ans_text = split_cot_answer(comp.text)
                try:
                    conf_letter, margin = answer_confidence(comp, valid)
                except Exception:
                    conf_letter, margin = None, None
                # Authoritative committed letter from the answer token; text-regex is the fallback.
                answer = conf_letter or parse_answer(ans_text, n_opt)
                flaky = margin is not None and margin < config.ANSWER_CONF_MARGIN
                samples.append(Sample(answer=answer, cot=cot, raw=comp.text,
                                      finish_reason=comp.finish_reason, answer_text=ans_text,
                                      conf_margin=margin, flaky=flaky))
            results.append(samples)
        return results
