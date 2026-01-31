from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def masked_softmax(logits, mask, axis=-1):
    mask = mask.astype(float)
    # set masked positions to large negative
    masked_logits = np.where(mask > 0, logits, -1e9)
    m = np.max(masked_logits, axis=axis, keepdims=True)
    ex = np.exp(masked_logits - m)
    ex = ex * mask
    denom = np.sum(ex, axis=axis, keepdims=True) + 1e-12
    return ex / denom


def soft_rank(x, tau: float):
    x = np.asarray(x)
    n = x.shape[0]
    ranks = np.ones_like(x, dtype=float)
    for i in range(n):
        diff = (x - x[i]) / tau
        ranks[i] = 1.0 + np.sum(sigmoid(diff)) - sigmoid(0.0)
    return ranks
