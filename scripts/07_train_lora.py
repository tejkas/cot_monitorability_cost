#!/usr/bin/env python
"""Phase 4b — LoRA-fine-tune Qwen3-8B to genuinely reason in T3 style (peft + trl SFT).

Trains a low-rank adapter (ΔW = (alpha/r)·B@A, base frozen) on the T3-style CoTs so the model
GENUINELY produces near-illegible reasoning. STYLE ONLY — the sway behavior emerges from the hint
at inference; it is NOT trained, and there is no monitor in the loop, so we don't induce
adversarial obfuscation (Baker's confound). The adapter is a small, swappable file.

Input:  data/traces/lora_sft.jsonl   (from 06)
Output: results/lora_t3/             (the LoRA adapter)

  python scripts/07_train_lora.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cotmon import config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap #SFT examples (smoke test)")
    ap.add_argument("--max-steps", type=int, default=0, help="cap training steps (smoke test)")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_dataset("json",
                      data_files=str(config.DATA_DIR / "traces" / "lora_sft.jsonl"),
                      split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[phase4b] {len(ds)} SFT examples"
          + (f" (smoke: max_steps={args.max_steps})" if args.max_steps else ""))

    lora = LoraConfig(
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES, task_type="CAUSAL_LM",
    )
    sft = SFTConfig(
        output_dir=str(config.RESULTS_DIR / config.LORA_OUT),
        num_train_epochs=config.LORA_EPOCHS,
        per_device_train_batch_size=config.LORA_BATCH,
        gradient_accumulation_steps=config.LORA_GRAD_ACCUM,
        learning_rate=config.LORA_LR,
        bf16=True, gradient_checkpointing=True,
        logging_steps=10, save_strategy="epoch", seed=config.SEED,
        max_length=config.MAX_MODEL_LEN,
        max_steps=args.max_steps if args.max_steps else -1,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=sft, train_dataset=ds,
                         peft_config=lora, processing_class=tok)
    trainer.train()
    trainer.save_model(str(config.RESULTS_DIR / config.LORA_OUT))
    print(f"[phase4b] LoRA adapter saved -> {config.RESULTS_DIR / config.LORA_OUT}")


if __name__ == "__main__":
    main()
