from __future__ import annotations

import numpy as np


def r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"r2": float("nan"), "rmse": float("nan")}
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    return {"r2": r2, "rmse": rmse}


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    # handle ties
    uniq, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    for u_idx, count in enumerate(counts):
        if count <= 1:
            continue
        idx = np.where(inv == u_idx)[0]
        ranks[idx] = ranks[idx].mean()
    return ranks


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.size == 0:
        return float("nan")
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(y_score)
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    if mask.sum() == 0:
        return float("nan")
    yt = y_true[mask]
    yp = y_score[mask]
    return float(np.mean((yt - yp) ** 2))
