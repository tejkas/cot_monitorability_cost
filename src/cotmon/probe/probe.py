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


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float = 0.01) -> float:
    """Recall at a threshold calibrated to `target_fpr` on the NEGATIVES.

    The deployment-relevant metric in both literatures this project sits between: probe papers
    report TPR@1%FPR alongside AUROC (McKenzie), and the CoT-monitoring papers calibrate at
    FPR<=1% and hold that operating point fixed (Zolkowski). AUROC alone hides whether a detector
    is usable at a tolerable alarm rate.
    """
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # smallest threshold whose FPR <= target: the (1-target) quantile of the negative scores
    thr = float(np.quantile(neg, 1.0 - target_fpr))
    return float(np.mean(pos > thr))


class LogRegProbe:
    """Logistic regression by gradient descent — the probe the literature actually uses.

    McKenzie/Meyoyan train linear read-outs with AdamW + BCE; none of the three probe papers use
    difference-of-means. We keep diff-of-means as primary (fewest parameters, best suited to our
    small-positive regime, cf. Marks & Tegmark) but report this alongside so the choice is visible
    rather than silently assumed. Plain numpy: no sklearn dependency.
    """

    def __init__(self, lr: float = 0.1, epochs: int = 300, l2: float = 1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogRegProbe":
        X = np.asarray(X, dtype=np.float32)
        self.mu, self.sigma = X.mean(axis=0), X.std(axis=0) + 1e-6
        Xs = (X - self.mu) / self.sigma
        n, d = Xs.shape
        self.w, self.b = np.zeros(d, dtype=np.float32), 0.0
        yf = y.astype(np.float32)
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-(Xs @ self.w + self.b)))
            g = (p - yf) / n
            self.w -= self.lr * (Xs.T @ g + self.l2 * self.w)
            self.b -= self.lr * float(g.sum())
        return self

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, dtype=np.float32) - self.mu) / self.sigma
        return Xs @ self.w + self.b


_PROBES = {"diff_of_means": DiffOfMeansProbe, "logreg": LogRegProbe}


def train_eval(X_tr: np.ndarray, y_tr: np.ndarray,
               X_te: np.ndarray, y_te: np.ndarray,
               kind: str = "diff_of_means") -> Dict[str, float]:
    """Fit `kind` on (X_tr, y_tr), score (X_te, y_te) -> auroc / tpr@1%fpr / acc."""
    if kind not in _PROBES:
        raise ValueError(f"probe kind {kind!r} not implemented; choose from {list(_PROBES)}")
    probe = _PROBES[kind]().fit(X_tr, y_tr)
    s_tr = probe.decision_scores(X_tr)
    s_te = probe.decision_scores(X_te)
    # accuracy needs a threshold: midpoint of the TRAIN class-score means (LDA-style boundary)
    thr = 0.5 * (s_tr[y_tr == 1].mean() + s_tr[y_tr == 0].mean())
    acc = float(((s_te > thr).astype(int) == y_te).mean())
    return {"auroc": auroc(y_te, s_te), "tpr@1fpr": tpr_at_fpr(y_te, s_te, 0.01), "acc": acc}
