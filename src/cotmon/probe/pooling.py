"""Token pooling for residual-stream activations (Phase 3).

Re-encoding a tier's text gives hidden states of shape [n_tokens, hidden_dim] at each layer.
A probe needs ONE vector per example, so we collapse the token axis. WHICH pooling best
surfaces the hint-usage signal is an open question — that's why we sweep over them
(McKenzie et al. Table 3): mean / max / last / softmax / attention.

Contract: every pooler takes `h` of shape [n_tokens, hidden_dim] (a float array) and returns
a vector of shape [hidden_dim]. `mask` (optional, [n_tokens] bool) marks real vs pad tokens.

    >>> vec = pool(h, "mean")          # vec.shape == (hidden_dim,)
"""
from typing import Optional

import numpy as np


def mean_pool(h: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    if mask is None:
        return np.mean(h, axis=0)
    return np.mean(h, axis=0, where=mask[:, None])


def max_pool(h: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    if mask is None:
        return np.max(h, axis=0)
    return np.max(h, axis=0, where=mask[:, None], initial=-np.inf)


def last_pool(h: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    if mask is None:
        return h[-1]
    return h[np.where(mask)[0][-1]]


def softmax_pool(h: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    scores = np.linalg.norm(h, axis=1) # (n_tokens,) L2 norm to get magnitude of feature vectors
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = np.exp(scores - scores.max())
    weights = weights/weights.sum()
    return weights @ h


def attention_pool(h: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    raise NotImplementedError("Defer for now.")


POOLERS = {
    "mean": mean_pool,
    "max": max_pool,
    "last": last_pool,
    "softmax": softmax_pool,
    #"attention": attention_pool,
}


def pool(h: np.ndarray, method: str, mask: Optional[np.ndarray] = None) -> np.ndarray:
    return POOLERS[method](h, mask)
