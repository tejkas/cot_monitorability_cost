"""Token pooling for residual-stream activations (Phase 3).

Re-encoding a tier's text gives hidden states of shape [n_tokens, hidden_dim] at each layer.
A probe needs ONE vector per example, so we collapse the token axis. WHICH pooling best
surfaces the hint-usage signal is an open question — that's why we sweep over them
(McKenzie et al. Table 3): mean / max / last / softmax / attention.

Contract: every pooler takes `h` of shape [n_tokens, hidden_dim] (a float array) and returns
a vector of shape [hidden_dim]. `mask` (optional, [n_tokens] bool) marks real vs pad tokens.

    >>> vec = pool(h, "mean")          # vec.shape == (hidden_dim,)

--- bodies are written by hand in education mode; interfaces are fixed here ---
"""
from typing import Optional

import numpy as np


def mean_pool(h: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    raise NotImplementedError("education mode")


def max_pool(h: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    raise NotImplementedError("education mode")


def last_pool(h: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    raise NotImplementedError("education mode")


def softmax_pool(h: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    raise NotImplementedError("education mode")


def attention_pool(h: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    raise NotImplementedError("education mode")


POOLERS = {
    "mean": mean_pool,
    "max": max_pool,
    "last": last_pool,
    "softmax": softmax_pool,
    "attention": attention_pool,
}


def pool(h: np.ndarray, method: str, mask: Optional[np.ndarray] = None) -> np.ndarray:
    return POOLERS[method](h, mask)
