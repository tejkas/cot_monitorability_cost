"""Deterministic tier-validation metrics (DESIGN §5) — no GPU / no model needed.

Two things we want to confirm about the rewriter's output:
  - readability DECREASES across tiers (a legibility gradient actually exists), and
  - keyword retention vs T0 DECREASES — i.e. the give-away content words get harder to
    surface as tiers compress. (If they survived intact a naive keyword monitor wouldn't
    degrade; if the text monitor degrades anyway, that's the interesting part.)

These are cheap proxies for a fast sanity check on the Mac. The definitive legibility test
is the Phase-3 text-monitor's actual accuracy per tier.
"""
import re
from typing import Dict, List

from .. import config

_WORD = re.compile(r"[A-Za-z']+")
_SYMBOL = re.compile(r"[^\w\s]")

# Common English function words: dense in prose (T0/T1), stripped in telegraphic/symbolic tiers.
_FUNCTION = set(
    """a an the of to in on at by for with and or but nor so yet if then else than that this
    these those is are was were be been being am do does did have has had having it its it's he
    she they them his her their we you i me my your our as from into over under about which who
    whom whose there here not no nor will would can could should shall may might must because
    while during between out up down off out's also very just only more most such own same""".split()
)


def _words(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text)]


def function_word_ratio(text: str) -> float:
    ws = _words(text)
    return sum(1 for w in ws if w in _FUNCTION) / len(ws) if ws else 0.0


def symbol_ratio(text: str) -> float:
    """Fraction of non-space characters that are punctuation/symbols (spikes in T3)."""
    dense = re.sub(r"\s", "", text)
    return len(_SYMBOL.findall(text)) / len(dense) if dense else 0.0


def readability(text: str) -> float:
    """Composite legibility proxy ~[0,1]: prose scores high, symbolic shorthand low.
    Function-word density (grammatical prose) discounted by symbol density."""
    return function_word_ratio(text) * (1.0 - symbol_ratio(text))


def content_words(text: str) -> set:
    return {w for w in _words(text) if w not in _FUNCTION and len(w) > 2}


def keyword_retention(tier_text: str, t0_text: str) -> float:
    """Fraction of T0's content words still present (as whole words) in the tier."""
    base = content_words(t0_text)
    return len(content_words(tier_text) & base) / len(base) if base else 0.0


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(tier_sets: List[Dict[str, str]]) -> dict:
    """Per-tier metric means across all traces + monotonicity verdicts."""
    tiers = config.TIER_NAMES
    acc = {t: {"readability": [], "fw_ratio": [], "symbol_ratio": [],
               "len_chars": [], "keyword_retention": []} for t in tiers}
    for ts in tier_sets:
        t0 = ts.get("T0_verbatim", "")
        for t in tiers:
            txt = ts.get(t, "")
            acc[t]["readability"].append(readability(txt))
            acc[t]["fw_ratio"].append(function_word_ratio(txt))
            acc[t]["symbol_ratio"].append(symbol_ratio(txt))
            acc[t]["len_chars"].append(len(txt))
            acc[t]["keyword_retention"].append(keyword_retention(txt, t0))

    per_tier = {t: {k: _mean(v) for k, v in d.items()} for t, d in acc.items()}

    def monotone_dec(seq: List[float]) -> bool:
        return all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))

    return {
        "n": len(tier_sets),
        "per_tier": per_tier,
        "readability_monotone": monotone_dec([per_tier[t]["readability"] for t in tiers]),
        "keyword_monotone": monotone_dec([per_tier[t]["keyword_retention"] for t in tiers]),
    }
