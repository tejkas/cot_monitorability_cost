"""Layer x pooling probe sweep (Phase 3).

For each (layer, pooling): train a probe on PROBE_TRAIN_TIER's activations and evaluate it on
EVERY tier's held-out activations -> accuracy/AUROC per tier. That's the probe side of the RQ1
divergence curve.

The split is BY QUESTION (not trace): sibling traces from one question must not straddle
train/test, or the probe can win by recognizing the question instead of the sway.
"""
import random
from typing import Dict, List, Sequence

import numpy as np

from .. import config
from . import probe as P


def question_split(question_ids: Sequence, test_fraction: float, seed: int):
    """Return (train_idx, test_idx) as arrays, splitting whole QUESTIONS into train/test."""
    qids = list(dict.fromkeys(question_ids))          # unique, order-preserving
    rng = random.Random(seed)
    rng.shuffle(qids)
    n_test = int(round(len(qids) * test_fraction))
    test_q = set(qids[:n_test])
    train_idx = [i for i, q in enumerate(question_ids) if q not in test_q]
    test_idx = [i for i, q in enumerate(question_ids) if q in test_q]
    return np.array(train_idx), np.array(test_idx)


def run_sweep(acts_by_tier: Dict[str, Dict[str, np.ndarray]],
              labels: np.ndarray, question_ids: Sequence,
              layers: List[int] = None, poolings: List[str] = None,
              tiers: List[str] = None, train_tier: str = None, kind: str = None) -> dict:
    """acts_by_tier[tier][pooling] has shape [n, n_layers, hidden]; the j-th layer slice == layers[j].

    Returns results[pooling][layer][tier] = {'auroc':..., 'acc':...}.
    """
    layers = layers or config.PROBE_LAYERS
    poolings = poolings or config.PROBE_POOLINGS
    tiers = tiers or config.TIER_NAMES
    train_tier = train_tier or config.PROBE_TRAIN_TIER
    kind = kind or config.PROBE_KIND

    tr, te = question_split(question_ids, config.PROBE_TEST_FRACTION, config.PROBE_SPLIT_SEED)
    y_tr, y_te = labels[tr], labels[te]

    results: dict = {}
    for pooling in poolings:
        results[pooling] = {}
        Xtr_all = acts_by_tier[train_tier][pooling]       # [n, n_layers, hidden]
        for j, layer in enumerate(layers):
            X_tr = Xtr_all[tr, j, :]
            results[pooling][layer] = {}
            for tier in tiers:
                X_te = acts_by_tier[tier][pooling][te, j, :]
                # NOTE: refits on the same X_tr each tier — cheap for diff-of-means; if we move
                # to logreg, fit once per (pooling, layer) and only re-score per tier.
                results[pooling][layer][tier] = P.train_eval(X_tr, y_tr, X_te, y_te, kind)
    return results
