from __future__ import annotations

from typing import Tuple

import numpy as np

from .tie import bottomk


def select_bottom2(scores: np.ndarray, active_mask: np.ndarray, tie_noise: np.ndarray | None = None) -> np.ndarray:
    return bottomk(scores, active_mask, 2, tie_noise=tie_noise)


def decide_save(J: np.ndarray, meanJ: np.ndarray, bottom2: np.ndarray,
                eta1: float, eta2: float, deterministic: bool, u: float) -> Tuple[int, int]:
    if bottom2.size < 2:
        return -1, -1
    i, j = int(bottom2[0]), int(bottom2[1])
    if deterministic:
        elim = i if J[i] < J[j] else j
        save = j if elim == i else i
        return elim, save
    diff_now = J[j] - J[i]
    diff_mean = meanJ[j] - meanJ[i]
    logit = eta1 * diff_now + eta2 * diff_mean
    p_elim_i = 1.0 / (1.0 + np.exp(-logit))
    if u < p_elim_i:
        return i, j
    return j, i
