from __future__ import annotations

import numpy as np


def masked_softmax(x: np.ndarray, mask: np.ndarray, axis: int = -1) -> np.ndarray:
    mask = mask.astype(bool)
    if x.shape != mask.shape:
        # broadcast mask
        mask = np.broadcast_to(mask, x.shape)
    x_masked = np.where(mask, x, -1e9)
    max_x = np.max(x_masked, axis=axis, keepdims=True)
    exp = np.exp(x_masked - max_x) * mask
    sum_exp = exp.sum(axis=axis, keepdims=True)
    return np.where(sum_exp > 0, exp / sum_exp, 0.0)


def compute_F_mode_a(theta: np.ndarray, active_mask: np.ndarray, center: bool = True) -> np.ndarray:
    if center:
        mask = active_mask.astype(bool)
        masked = np.where(mask, theta, 0.0)
        count = mask.sum(axis=-1, keepdims=True)
        mean = np.where(count > 0, masked.sum(axis=-1, keepdims=True) / count, 0.0)
        theta = theta - mean
    return masked_softmax(theta, active_mask, axis=-1)
