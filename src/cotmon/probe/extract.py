"""Residual-stream activation extraction via HF Transformers (Phase 3).

Re-encode each tier's TEXT through the base model with `output_hidden_states=True`, then pool
over tokens -> one vector per (example, layer, pooling). This is the RE-ENCODE probe input
(DESIGN §6): the SAME tier-k text the text monitor reads, read as activations instead of tokens.

Done in HF Transformers, NOT vLLM, on purpose: vLLM hides per-token hidden states behind its
paged-attention kernels; forcing them out during decode is exactly the cost the Phase-5 serving
study measures. Here (offline, prefill-only) HF is the clean tool.

Output of extract(): {pooling: np.ndarray[n_texts, n_layers, hidden]}, where the j-th layer
slice corresponds to config.PROBE_LAYERS[j]. The pad-aware attention mask is passed to each
pooler so padding tokens never contribute.

torch/transformers are imported lazily so this module imports on a laptop for syntax checks.
"""
import time
from typing import Dict, List, Optional

import numpy as np

from .. import config
from .pooling import pool


class ActivationExtractor:
    """Loads the base model in HF with hidden-state output; re-encodes + pools text batches."""

    def __init__(self, model: str = config.BASE_MODEL, layers: Optional[List[int]] = None,
                 poolings: Optional[List[str]] = None,
                 batch_size: int = config.EXTRACT_BATCH_SIZE,
                 max_len: int = config.EXTRACT_MAX_LEN,
                 lora: Optional[str] = None):
        import torch  # lazy
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.layers = layers or config.PROBE_LAYERS
        self.poolings = poolings or config.PROBE_POOLINGS
        self.batch_size = batch_size
        self.max_len = max_len

        self.tok = AutoTokenizer.from_pretrained(model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # AutoModel, NOT AutoModelForCausalLM: the backbone has no lm_head, so it never computes
        # the [B, S, vocab] logits (~9 GiB at batch 8 / seq 4k) we'd only throw away. We want
        # hidden states, not predictions.
        if lora:
            # Probe the FINE-TUNED model, not the base one. When the LoRA'd model authored the
            # text, its own activations are the right probe target — base-model activations on
            # out-of-distribution LoRA notation are a different (and weaker) signal.
            # merge_and_unload folds the adapter into the weights, then .model drops the lm_head.
            from peft import PeftModel
            from transformers import AutoModelForCausalLM
            m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
            m = PeftModel.from_pretrained(m, lora)
            self.model = m.merge_and_unload().model.to(self.device).eval()
            print(f"[extract] probing the LoRA-merged model (adapter: {lora})")
        else:
            self.model = AutoModel.from_pretrained(
                model, torch_dtype=torch.float16, output_hidden_states=True,
            ).to(self.device).eval()
        self.hidden = self.model.config.hidden_size

    def chat_prompt(self, user_turn: str) -> str:
        """Chat-templated prompt string, matching what the model saw at generation time.

        Teacher-forcing only reproduces generation-time activations if the prefix is the SAME
        prefix the model actually conditioned on — so the prompt must go through the chat
        template exactly as data/generate.Generator._template does it.
        """
        try:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": user_turn}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
        except TypeError:  # template without the thinking kwarg
            return self.tok.apply_chat_template(
                [{"role": "user", "content": user_turn}],
                tokenize=False, add_generation_prompt=True)

    def extract(self, texts: List[str],
                prompts: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Return {pooling: [n_texts, n_layers, hidden]} (float32).

        prompts is the CORRECT mode and should always be supplied: the forward pass covers
        `prompt + text`, but pooling is restricted to the TEXT (completion) positions. Because
        attention is causal, a teacher-forced pass over prompt+completion reproduces the model's
        generation-time hidden states exactly — so this measures the state the model was in while
        AUTHORING, with the question and hint in context.

        This matches the only published precedent for probing a property of a model's own response
        (Aremu et al.: teacher-forced prompt+completion at train time, pooling over assistant
        tokens only, prompt always in context). NO paper feeds the response in isolation.

        prompts=None reproduces the old prompt-free behaviour and is kept only for reproducing
        earlier results — it asks the probe to detect hint-deference from activations that never
        saw the hint. It warns loudly.
        """
        torch = self.torch
        n = len(texts)
        if prompts is None:
            print("[extract] WARNING: no prompts given -> pooling over the completion ALONE. "
                  "The model never sees the question/hint; this has no precedent in the "
                  "literature and understates recoverable signal. Pass prompts=...", flush=True)
        elif len(prompts) != n:
            raise ValueError(f"prompts ({len(prompts)}) and texts ({n}) must be the same length")

        out = {p: np.empty((n, len(self.layers), self.hidden), dtype=np.float32)
               for p in self.poolings}

        n_batches = (n + self.batch_size - 1) // self.batch_size
        print(f"[extract] {n} texts on {self.device} -> {n_batches} batches of {self.batch_size} "
              f"(max_len={self.max_len}, "
              f"context={'prompt+completion' if prompts else 'completion only'})",
              flush=True)  # flush: stdout is buffered under nohup/redirect; without this you go blind
        t0 = time.time()
        self.tok.padding_side = "right"  # keeps prompt-length offsets valid per row
        for bidx, start in enumerate(range(0, n, self.batch_size)):
            batch = texts[start:start + self.batch_size]
            if prompts is None:
                seqs, plens = batch, [0] * len(batch)
            else:
                pbatch = prompts[start:start + self.batch_size]
                seqs = [p + t for p, t in zip(pbatch, batch)]
                # token length of each prompt -> where the completion span begins.
                # add_special_tokens=False everywhere so the two tokenizations stay aligned.
                plens = [len(self.tok(p, add_special_tokens=False)["input_ids"]) for p in pbatch]
            enc = self.tok(seqs, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_len,
                           add_special_tokens=prompts is None).to(self.device)
            with torch.no_grad():
                # request hidden states explicitly (the LoRA-merged backbone isn't loaded with
                # output_hidden_states baked into its config)
                hs = self.model(**enc, output_hidden_states=True).hidden_states  # (n_layers+1) x [B,S,H]

            masks = enc["attention_mask"].bool().cpu().numpy()          # [B, S]
            # Pull only the layers we need to host memory once (hs[0] = embeddings, hs[L] = block L).
            layer_arrs = [hs[L].float().cpu().numpy() for L in self.layers]  # each [B, S, hidden]

            for bi in range(len(batch)):
                m = masks[bi].copy()  # [S] bool — real vs pad tokens
                if plens[bi]:
                    # Drop PROMPT positions from pooling: the prompt is in context (it conditions
                    # every completion activation via causal attention) but must not be pooled,
                    # or the probe could score on question identity alone.
                    m[:plens[bi]] = False
                if not m.any():
                    # empty/blank completion (or fully truncated) -> all-pad row; last_pool would
                    # IndexError. Fall back to the real tokens so we emit a finite vector.
                    m = masks[bi].copy()
                if not m.any():
                    m = np.ones_like(m)
                for p in self.poolings:
                    for j in range(len(self.layers)):
                        out[p][start + bi, j] = pool(layer_arrs[j][bi], p, mask=m)

            del hs, layer_arrs
            if self.device == "cuda":
                torch.cuda.empty_cache()

            if (bidx + 1) % 10 == 0 or bidx == n_batches - 1:
                done = min(start + self.batch_size, n)
                el = time.time() - t0
                rate = done / el if el else 0.0
                eta = (n - done) / rate / 60 if rate else 0.0
                print(f"[extract] batch {bidx + 1}/{n_batches}  ({done}/{n} texts, "
                      f"{rate:.1f} txt/s, ETA {eta:.1f} min)", flush=True)
        return out
