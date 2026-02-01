from __future__ import annotations

import numpy as np


def rankdata(values: np.ndarray, ascending: bool = True) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values if ascending else -values)
    ranks = np.empty_like(values, dtype=float)
    n = len(values)
    i = 0
    rank = 1
    while i < n:
        j = i
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        avg = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        rank += (j - i)
        i = j
    return ranks


def compute_elim_week(elim_matrix: np.ndarray, active_init: np.ndarray) -> np.ndarray:
    # elim_matrix: [T, N]
    T, N = elim_matrix.shape
    elim_week = np.full((N,), fill_value=T + 1, dtype=int)
    for i in range(N):
        if not active_init[i]:
            elim_week[i] = 0
            continue
        weeks = np.where(elim_matrix[:, i])[0]
        if weeks.size > 0:
            elim_week[i] = int(weeks.min()) + 1
    return elim_week


def compute_final_rank(elim_matrix: np.ndarray, active_init: np.ndarray) -> np.ndarray:
    # returns rank 1..N for active contestants, 0 for padded
    # The contestant(s) who survive the longest get rank 1 (winner)
    elim_week = compute_elim_week(elim_matrix, active_init)
    N = elim_week.shape[0]
    rank = np.zeros((N,), dtype=float)
    active_idx = np.where(active_init)[0]
    if active_idx.size == 0:
        return rank
    # higher elim_week -> better rank (rank 1 = winner)
    active_weeks = elim_week[active_idx]
    # Use descending order: highest elim_week gets rank 1
    order = np.argsort(-active_weeks)  # descending
    n = len(active_weeks)
    ranks = np.zeros(n, dtype=float)
    i = 0
    current_rank = 1
    while i < n:
        j = i
        # Find all tied contestants
        while j < n and active_weeks[order[j]] == active_weeks[order[i]]:
            j += 1
        # Assign average rank for ties
        avg_rank = (current_rank + (current_rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        current_rank += (j - i)
        i = j
    rank[active_idx] = ranks
    return rank


def compute_survival_time(active_mask: np.ndarray) -> np.ndarray:
    # active_mask: [T, N]
    return active_mask.sum(axis=0).astype(int)
