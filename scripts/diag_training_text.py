#!/usr/bin/env python
"""Diagnostic A (CPU, cheapest): what did trl actually train on?

Apply the chat template to a lora_sft.jsonl example the way trl does, and check whether the
symbolic <think> content survived — Qwen3's template is known to STRIP <think>...</think> from
assistant messages in some conditions. If it stripped, the model only ever saw "Final answer:(X)"
as the target and never learned the symbolic style -> that's the root cause (and a clean fix).

  python scripts/diag_training_text.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    ex = json.loads(open(config.DATA_DIR / "traces" / "lora_sft.jsonl").readline())
    text = tok.apply_chat_template(ex["messages"], tokenize=False)

    print("=== rendered training text (what trl builds from the messages) ===")
    print(text)
    print("=" * 80)
    print("contains '<think>'        :", "<think>" in text)
    print("contains symbolic content :", ("Ans=" in text) or ("→" in text) or ("|" in text))
    print("rendered length (chars)   :", len(text))
    # If '<think>' + symbolic content are BOTH present, the style was in the training tokens.
    # If they're absent (assistant reduced to 'Final answer: (X)'), the template stripped it.


if __name__ == "__main__":
    main()
