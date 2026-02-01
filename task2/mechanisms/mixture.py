from __future__ import annotations

import numpy as np


def _select_gumbel(gumbel_noise: np.ndarray | None, idx: np.ndarray) -> np.ndarray:
    if gumbel_noise is None:
        return np.random.gumbel(size=idx.shape[0])
    g = np.asarray(gumbel_noise)
    if g.shape[0] == idx.shape[0]:
        return g
    return g[idx]


def soft_probs_percent(C: np.ndarray, active_mask: np.ndarray, kappa: float) -> np.ndarray:
    mask = active_mask.astype(bool)
    vals = np.where(mask, C, 0.0)
    p = np.exp(-kappa * vals) * mask
    denom = p.sum()
    if denom <= 0:
        return np.where(mask, 1.0, 0.0)
    return p / denom


def soft_probs_rank(R: np.ndarray, active_mask: np.ndarray, kappa_r: float) -> np.ndarray:
    mask = active_mask.astype(bool)
    vals = np.where(mask, R, 0.0)
    p = np.exp(kappa_r * vals) * mask
    denom = p.sum()
    if denom <= 0:
        return np.where(mask, 1.0, 0.0)
    return p / denom


def mix_probs(pP: np.ndarray, pR: np.ndarray, w: float) -> np.ndarray:
    p = (pP ** w) * (pR ** (1.0 - w))
    denom = p.sum()
    if denom <= 0:
        return p
    return p / denom


def select_elim_mix(p: np.ndarray, active_mask: np.ndarray, d: int, gumbel_noise: np.ndarray | None = None) -> np.ndarray:
    mask = active_mask.astype(bool)
    idx = np.where(mask)[0]
    if idx.size == 0 or d <= 0:
        return np.array([], dtype=int)
    g = _select_gumbel(gumbel_noise, idx)
    scores = np.log(p[idx] + 1e-12) + g
    order = np.argsort(-scores)
    return idx[order[:d]]
