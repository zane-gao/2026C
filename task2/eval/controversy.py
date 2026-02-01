from __future__ import annotations

import numpy as np


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    # values shape [N]
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    n = len(values)
    ranks[order] = np.arange(1, n + 1)
    return ranks / n


def compute_controversy_group(S_bar: np.ndarray, F_bar: np.ndarray, q: float = 0.2) -> np.ndarray:
    # S_bar, F_bar: [K, S, N]
    S_mean = S_bar.mean(axis=0)
    F_mean = F_bar.mean(axis=0)
    S, N = S_mean.shape
    group = np.zeros((S, N), dtype=bool)
    for s in range(S):
        mask = np.isfinite(S_mean[s]) & np.isfinite(F_mean[s])
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        pS = _percentile_ranks(S_mean[s, idx])
        pF = _percentile_ranks(F_mean[s, idx])
        ci = pF - pS
        cutoff = np.quantile(ci, 1.0 - q) if ci.size > 0 else 1.0
        group_idx = idx[ci >= cutoff]
        group[s, group_idx] = True
    return group
