from __future__ import annotations

import numpy as np

from .tie import rank_desc, topk


def _select_gumbel(gumbel_noise: np.ndarray | None, idx: np.ndarray) -> np.ndarray:
    if gumbel_noise is None:
        return np.random.gumbel(size=idx.shape[0])
    g = np.asarray(gumbel_noise)
    if g.shape[0] == idx.shape[0]:
        return g
    return g[idx]


def compute_rank_scores(J: np.ndarray, F: np.ndarray, active_mask: np.ndarray, tie_mode: str = "average",
                        tie_noise: np.ndarray | None = None, alpha: float = 1e-6) -> np.ndarray:
    rJ = rank_desc(J, active_mask, tie_mode=tie_mode, tie_noise=tie_noise)
    rF = rank_desc(F, active_mask, tie_mode=tie_mode, tie_noise=tie_noise)
    R = rJ + rF + alpha * rF
    return R


def select_elim_rank(R: np.ndarray, active_mask: np.ndarray, d: int, tie_noise: np.ndarray | None = None,
                     soft: bool = False, gumbel_noise: np.ndarray | None = None, kappa_r: float = 10.0) -> np.ndarray:
    mask = active_mask.astype(bool)
    if d <= 0:
        return np.array([], dtype=int)
    if not soft:
        return topk(R, mask, d, tie_noise=tie_noise)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.array([], dtype=int)
    p = np.exp(kappa_r * R[idx])
    p = p / np.clip(p.sum(), 1e-12, None)
    g = _select_gumbel(gumbel_noise, idx)
    scores = np.log(p + 1e-12) + g
    order = np.argsort(-scores)
    return idx[order[:d]]


def margin_rank(R: np.ndarray, active_mask: np.ndarray, d: int) -> float:
    if d <= 0:
        return float('nan')
    idx = np.where(active_mask)[0]
    if idx.size <= d:
        return float('nan')
    vals = np.sort(R[idx])[::-1]
    if d >= len(vals):
        return float('nan')
    return float(vals[d - 1] - vals[d])
