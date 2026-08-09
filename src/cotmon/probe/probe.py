"""Activation probes (Phase 3): difference-of-means (primary) + logistic regression.

Given pooled activation vectors X of shape [n, hidden_dim] and binary labels y (1 = swayed,
0 = resisted), fit a probe and score held-out data.

Difference-of-means: direction w = mean(X[y==1]) - mean(X[y==0]); an example's score is its
projection X @ w (optionally on normalized activations). Simple, robust, hard to overfit, and
interpretable — the right first probe before anything fancier. `logreg` is the alternative.

`train_eval` returns a metrics dict, at least {"auroc": float, "acc": float}.
"""
from typing import Dict

import numpy as np


class DiffOfMeansProbe:
    """Direction = class-mean difference; decision score = projection onto it."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DiffOfMeansProbe":
        # Activations are cached as fp16; summing ~thousands of rows in a float16 accumulator
        # overflows its +-65504 range -> inf/NaN. Upcast to float32 before any reduction.
        X = np.asarray(X, dtype=np.float32)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-6
        # standardize to z-scores
        Xs = (X - self.mu) / self.sigma
        self.w = Xs[y == 1].mean(axis=0) - Xs[y == 0].mean(axis=0)
        return self

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, dtype=np.float32) - self.mu) / self.sigma
        return Xs @ self.w

def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """P(a random positive scores higher than a random negative); ties = 0.5. Threshold-free."""
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg)))


def train_eval(X_tr: np.ndarray, y_tr: np.ndarray,
               X_te: np.ndarray, y_te: np.ndarray,
               kind: str = "diff_of_means") -> Dict[str, float]:
    """Fit `kind` on (X_tr, y_tr), score (X_te, y_te) -> {'auroc':..., 'acc':...}."""
    if kind != "diff_of_means":
        raise ValueError(f"probe kind {kind!r} not implemented (logreg deferred)")
    probe = DiffOfMeansProbe().fit(X_tr, y_tr)
    s_tr = probe.decision_scores(X_tr)
    s_te = probe.decision_scores(X_te)
    # accuracy needs a threshold: midpoint of the TRAIN class-score means (LDA-style boundary)
    thr = 0.5 * (s_tr[y_tr == 1].mean() + s_tr[y_tr == 0].mean())
    acc = float(((s_te > thr).astype(int) == y_te).mean())
    return {"auroc": auroc(y_te, s_te), "acc": acc}
