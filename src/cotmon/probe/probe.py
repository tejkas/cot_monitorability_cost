"""Activation probes (Phase 3): difference-of-means (primary) + logistic regression.

Given pooled activation vectors X of shape [n, hidden_dim] and binary labels y (1 = swayed,
0 = resisted), fit a probe and score held-out data.

Difference-of-means: direction w = mean(X[y==1]) - mean(X[y==0]); an example's score is its
projection X @ w (optionally on normalized activations). Simple, robust, hard to overfit, and
interpretable — the right first probe before anything fancier. `logreg` is the alternative.

`train_eval` returns a metrics dict, at least {"auroc": float, "acc": float}.

--- bodies are written by hand in education mode; interfaces are fixed here ---
"""
from typing import Dict

import numpy as np


class DiffOfMeansProbe:
    """Direction = class-mean difference; decision score = projection onto it."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DiffOfMeansProbe":
        raise NotImplementedError("education mode")

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("education mode")


def train_eval(X_tr: np.ndarray, y_tr: np.ndarray,
               X_te: np.ndarray, y_te: np.ndarray,
               kind: str = "diff_of_means") -> Dict[str, float]:
    """Fit `kind` on (X_tr, y_tr), score (X_te, y_te) -> {'auroc':..., 'acc':...}."""
    raise NotImplementedError("education mode")
