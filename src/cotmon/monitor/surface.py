"""TF-IDF surface baseline — the cheap-lexical control the probe literature expects.

McKenzie et al. run a TF-IDF+SVM word-statistics baseline on every evaluation set, and separately
find that activation probes DO fire on surface keywords (they are not surface-invariant). So
"is the probe reading the reasoning, or just lexical cues?" has to be measured, not assumed.

Any AUROC this baseline achieves is attributable to word statistics alone. The probe and the
monitor must beat it to claim they are doing something more than keyword matching.

Hand-rolled (no sklearn dependency): term counts -> tf-idf -> the same LogRegProbe used elsewhere,
so the classifier is held constant and only the FEATURES differ.
"""
import re
from typing import Dict, List

import numpy as np

from ..probe.probe import LogRegProbe, auroc, tpr_at_fpr

_TOKEN = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class TfidfSurfaceBaseline:
    """Bag-of-words tf-idf + logistic regression over the CoT text."""

    def __init__(self, max_features: int = 5000, min_df: int = 2):
        self.max_features, self.min_df = max_features, min_df

    def _fit_vocab(self, docs: List[List[str]]) -> None:
        df: Dict[str, int] = {}
        for toks in docs:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        keep = [(w, c) for w, c in df.items() if c >= self.min_df]
        keep.sort(key=lambda wc: (-wc[1], wc[0]))
        keep = keep[:self.max_features]
        self.vocab = {w: i for i, (w, _) in enumerate(keep)}
        n = len(docs)
        self.idf = np.ones(len(self.vocab), dtype=np.float32)
        for w, i in self.vocab.items():
            self.idf[i] = np.log((1 + n) / (1 + df[w])) + 1.0

    def _vectorize(self, docs: List[List[str]]) -> np.ndarray:
        X = np.zeros((len(docs), len(self.vocab)), dtype=np.float32)
        for r, toks in enumerate(docs):
            for w in toks:
                i = self.vocab.get(w)
                if i is not None:
                    X[r, i] += 1.0
        X *= self.idf                                  # tf-idf
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-8)             # L2 normalize

    def fit(self, texts: List[str], y: np.ndarray) -> "TfidfSurfaceBaseline":
        docs = [_tokenize(t) for t in texts]
        self._fit_vocab(docs)
        self.clf = LogRegProbe(lr=0.5, epochs=400, l2=1e-3).fit(self._vectorize(docs), y)
        return self

    def scores(self, texts: List[str]) -> np.ndarray:
        return self.clf.decision_scores(self._vectorize([_tokenize(t) for t in texts]))


def train_eval_surface(texts_tr: List[str], y_tr: np.ndarray,
                       texts_te: List[str], y_te: np.ndarray) -> Dict[str, float]:
    """Fit the surface baseline on train text, score test text -> auroc / tpr@1%fpr."""
    b = TfidfSurfaceBaseline().fit(texts_tr, y_tr)
    s = b.scores(texts_te)
    return {"auroc": auroc(y_te, s), "tpr@1fpr": tpr_at_fpr(y_te, s, 0.01),
            "n_features": len(b.vocab)}
