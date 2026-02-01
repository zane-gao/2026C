from __future__ import annotations

import numpy as np


def generate_linear_samples(a: np.ndarray, b: np.ndarray, alpha: np.ndarray, sigma_J: float, K: int, rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng()
    S, N = a.shape
    T = alpha.shape[1]
    t_idx = (np.arange(T) + 1).astype(float)
    base = a[:, None, :] + b[:, None, :] * t_idx[None, :, None]
    base = np.broadcast_to(base, (S, T, N))
    alpha_st = alpha[:, :, None]
    S_samples = np.broadcast_to(base, (K, S, T, N)).astype(float)
    eps = rng.normal(loc=0.0, scale=sigma_J, size=(K, S, T, N)).astype(float)
    tildeJ = alpha_st[None, ...] + S_samples + eps
    return S_samples, tildeJ, eps


def generate_rw_samples(a: np.ndarray, alpha: np.ndarray, sigma_rw: float, sigma_J: float, K: int,
                        rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng()
    S, N = a.shape
    T = alpha.shape[1]
    S_samples = np.zeros((K, S, T, N), dtype=float)
    eps = rng.normal(loc=0.0, scale=sigma_J, size=(K, S, T, N)).astype(float)
    for k in range(K):
        S_samples[k, :, 0, :] = a
        for t in range(1, T):
            step = rng.normal(loc=0.0, scale=sigma_rw, size=(S, N))
            S_samples[k, :, t, :] = S_samples[k, :, t - 1, :] + step
    tildeJ = alpha[:, :, None][None, ...] + S_samples + eps
    return S_samples, tildeJ, eps
