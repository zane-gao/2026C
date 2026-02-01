from __future__ import annotations

import numpy as np

from .mode_a import masked_softmax


def compute_theta_mode_b(mu: np.ndarray, gamma: np.ndarray, epsilon: np.ndarray, J_z: np.ndarray) -> np.ndarray:
    # broadcast shapes
    theta = mu + gamma[..., None, None] * J_z + epsilon
    return theta


def compute_F_mode_b(mu: np.ndarray, gamma: np.ndarray, epsilon: np.ndarray, J_z: np.ndarray, active_mask: np.ndarray,
                     center: bool = True) -> np.ndarray:
    theta = compute_theta_mode_b(mu, gamma, epsilon, J_z)
    if center:
        mask = active_mask.astype(bool)
        masked = np.where(mask, theta, 0.0)
        count = mask.sum(axis=-1, keepdims=True)
        mean = np.where(count > 0, masked.sum(axis=-1, keepdims=True) / count, 0.0)
        theta = theta - mean
    return masked_softmax(theta, active_mask, axis=-1)
