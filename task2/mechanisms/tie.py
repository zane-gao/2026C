from __future__ import annotations

import numpy as np


def rank_desc(values: np.ndarray, mask: np.ndarray | None = None, tie_mode: str = "average",
              tie_noise: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if mask is None:
        mask = np.ones_like(values, dtype=bool)
    mask = mask.astype(bool)
    out = np.zeros_like(values, dtype=float)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return out
    vals = values[idx]
    if tie_mode == "random" and tie_noise is not None:
        vals = vals + 1e-6 * tie_noise[idx]
        order = np.argsort(-vals)
        ranks = np.empty_like(vals, dtype=float)
        ranks[order] = np.arange(1, len(vals) + 1)
    else:
        order = np.argsort(-vals)
        ranks = np.empty_like(vals, dtype=float)
        i = 0
        rank = 1
        while i < len(vals):
            j = i
            while j < len(vals) and vals[order[j]] == vals[order[i]]:
                j += 1
            avg = (rank + (rank + (j - i) - 1)) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg
            rank += (j - i)
            i = j
    out[idx] = ranks
    return out


def bottomk(values: np.ndarray, mask: np.ndarray, k: int, tie_noise: np.ndarray | None = None) -> np.ndarray:
    idx = np.where(mask)[0]
    if idx.size == 0 or k <= 0:
        return np.array([], dtype=int)
    vals = values[idx].astype(float)
    if tie_noise is not None:
        vals = vals + 1e-6 * tie_noise[idx]
    order = np.argsort(vals)
    return idx[order[:k]]


def topk(values: np.ndarray, mask: np.ndarray, k: int, tie_noise: np.ndarray | None = None) -> np.ndarray:
    idx = np.where(mask)[0]
    if idx.size == 0 or k <= 0:
        return np.array([], dtype=int)
    vals = values[idx].astype(float)
    if tie_noise is not None:
        vals = vals + 1e-6 * tie_noise[idx]
    order = np.argsort(-vals)
    return idx[order[:k]]
