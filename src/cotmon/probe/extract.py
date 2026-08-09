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
from typing import Dict, List, Optional

import numpy as np

from .. import config
from .pooling import pool


class ActivationExtractor:
    """Loads the base model in HF with hidden-state output; re-encodes + pools text batches."""

    def __init__(self, model: str = config.BASE_MODEL, layers: Optional[List[int]] = None,
                 poolings: Optional[List[str]] = None,
                 batch_size: int = config.EXTRACT_BATCH_SIZE,
                 max_len: int = config.EXTRACT_MAX_LEN):
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
        self.model = AutoModel.from_pretrained(
            model, torch_dtype=torch.float16, output_hidden_states=True,
        ).to(self.device).eval()
        self.hidden = self.model.config.hidden_size

    def extract(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Re-encode `texts`; return {pooling: [n_texts, n_layers, hidden]} (float32)."""
        torch = self.torch
        n = len(texts)
        out = {p: np.empty((n, len(self.layers), self.hidden), dtype=np.float32)
               for p in self.poolings}

        for start in range(0, n, self.batch_size):
            batch = texts[start:start + self.batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=self.max_len).to(self.device)
            with torch.no_grad():
                hs = self.model(**enc).hidden_states  # tuple (n_layers+1) of [B, S, hidden]

            masks = enc["attention_mask"].bool().cpu().numpy()          # [B, S]
            # Pull only the layers we need to host memory once (hs[0] = embeddings, hs[L] = block L).
            layer_arrs = [hs[L].float().cpu().numpy() for L in self.layers]  # each [B, S, hidden]

            for bi in range(len(batch)):
                m = masks[bi]  # [S] bool — real vs pad tokens
                for p in self.poolings:
                    for j in range(len(self.layers)):
                        out[p][start + bi, j] = pool(layer_arrs[j][bi], p, mask=m)

            del hs, layer_arrs
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return out
