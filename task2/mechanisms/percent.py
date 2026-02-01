from __future__ import annotations

import numpy as np

from .tie import bottomk


def _select_gumbel(gumbel_noise: np.ndarray | None, idx: np.ndarray) -> np.ndarray:
    if gumbel_noise is None:
        return np.random.gumbel(size=idx.shape[0])
    g = np.asarray(gumbel_noise)
    if g.shape[0] == idx.shape[0]:
        return g
    return g[idx]


def compute_percent_scores(J: np.ndarray, F: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    mask = active_mask.astype(bool)
    J_masked = np.where(mask, J, 0.0)
    denom = J_masked.sum(axis=-1, keepdims=True)
    q = np.where(denom > 0, J_masked / denom, 0.0)
    C = q + F
    return C


def select_elim_percent(C: np.ndarray, active_mask: np.ndarray, d: int, tie_noise: np.ndarray | None = None,
                        soft: bool = False, gumbel_noise: np.ndarray | None = None, kappa: float = 20.0) -> np.ndarray:
    mask = active_mask.astype(bool)
    if d <= 0:
        return np.array([], dtype=int)
    if not soft:
        return bottomk(C, mask, d, tie_noise=tie_noise)
    # soft without replacement via gumbel-top-k
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.array([], dtype=int)
    p = np.exp(-kappa * C[idx])
    p = p / np.clip(p.sum(), 1e-12, None)
    g = _select_gumbel(gumbel_noise, idx)
    scores = np.log(p + 1e-12) + g
    order = np.argsort(-scores)
    return idx[order[:d]]


def margin_percent(C: np.ndarray, active_mask: np.ndarray, d: int) -> float:
    if d <= 0:
        return float('nan')
    idx = np.where(active_mask)[0]
    if idx.size <= d:
        return float('nan')
    vals = np.sort(C[idx])
    if d >= len(vals):
        return float('nan')
    return float(vals[d] - vals[d - 1])
