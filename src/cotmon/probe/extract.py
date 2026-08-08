"""Residual-stream activation extraction via HF Transformers (Phase 3).

Re-encode each tier's TEXT through the base model with `output_hidden_states=True`, then pool
over tokens -> one vector per (example, layer, pooling). This produces the RE-ENCODE probe
input (DESIGN §6): the SAME tier-k text the text monitor reads, but read as activations.

Done in HF Transformers, NOT vLLM, on purpose: vLLM hides per-token hidden states behind its
paged-attention kernels, and forcing them out during decode is exactly the cost the Phase-5
serving study measures. Here (offline, prefill-only re-encoding) HF is the clean tool.

Activations are large — cache to config.ACTIVATIONS_DIR as fp16, per (tier, pooling):
    X[tier][pooling] : np.ndarray [n_examples, n_layers, hidden_dim]
    plus aligned labels [n] and question_ids [n] for a leakage-safe split.

--- the extraction internals (residual stream, output_hidden_states, HF-vs-vLLM) are covered
    in education mode; this file fixes the interface + caching plumbing. ---
"""
from typing import Dict, List

from .. import config


class ActivationExtractor:
    """Loads the base model in HF with hidden-state output enabled; re-encodes + pools text."""

    def __init__(self, model: str = config.BASE_MODEL,
                 layers: List[int] = None, poolings: List[str] = None):
        self.model_name = model
        self.layers = layers or config.PROBE_LAYERS
        self.poolings = poolings or config.PROBE_POOLINGS
        raise NotImplementedError("education mode (HF load + output_hidden_states)")

    def extract(self, texts: List[str]) -> Dict[str, "object"]:
        """Re-encode `texts`, pool per layer x pooling.

        Returns {pooling: array[n_texts, n_layers, hidden_dim]} (n_layers == len(self.layers)).
        """
        raise NotImplementedError("education mode (forward pass + pooling)")
