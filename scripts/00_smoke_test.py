#!/usr/bin/env python
"""Phase 0 smoke test — prove the plumbing before investing in Phase 1.

Confirms, on whatever device is available:
  1. The base model loads and generates.
  2. Thinking mode produces a <think>...</think> block.
  3. We can extract residual-stream activations at a chosen layer (the HF path).

  python scripts/00_smoke_test.py                          # Qwen3-8B (GPU box)
  python scripts/00_smoke_test.py --model Qwen/Qwen3-0.6B  # quick local code-path check

NOTE: the activation grab here is a throwaway shape-check. The real extraction design
(where in the block we tap, how we pool, re-encode vs. genuine) is a Phase-3 topic.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

QUESTION = (
    "A ball is thrown straight up. At the highest point of its motion, which is true?\n"
    "(A) velocity zero, acceleration zero\n"
    "(B) velocity zero, acceleration nonzero\n"
    "(C) velocity nonzero, acceleration zero\n"
    "(D) velocity nonzero, acceleration nonzero\n"
    "Answer with the letter."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.BASE_MODEL)
    ap.add_argument("--layer", type=int, default=config.PROBE_LAYERS[len(config.PROBE_LAYERS) // 2])
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    device = config.get_device()
    print(f"[smoke] device={device}  model={args.model}  probe_layer={args.layer}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device)
    model.eval()

    # --- 1 & 2: generate with thinking enabled ---
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Qwen3 hybrid-thinking toggle
    )
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    gen = out[0][inputs.input_ids.shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True)
    # Qwen3-4B-Thinking-2507 may emit only the closing tag — accept either.
    has_think = "</think>" in text or "<think>" in text
    print(f"[smoke] generated {gen.shape[0]} tokens; thinking_block={'YES' if has_think else 'NO'}")
    print("[smoke] --- output (truncated) ---")
    print(text[:800])

    # --- 3: extract residual-stream activations (throwaway shape-check) ---
    with torch.no_grad():
        hs = model(out, output_hidden_states=True).hidden_states
    # hidden_states is a tuple of length num_layers+1: embeddings, then each block's output.
    act = hs[args.layer]  # (batch, seq_len, hidden)
    print(f"[smoke] hidden_states tuple len={len(hs)} (embeddings + {len(hs) - 1} layers)")
    print(f"[smoke] layer {args.layer} activation shape={tuple(act.shape)} "
          f"(expect (1, seq_len, {model.config.hidden_size}))")

    ok = has_think and act.shape[-1] == model.config.hidden_size
    print(f"[smoke] {'PASS' if ok else 'FAIL'} — plumbing {'works' if ok else 'needs attention'}.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
