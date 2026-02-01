from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_jmax(J_obs: np.ndarray, active_mask: np.ndarray, policy: str = "max_obs") -> np.ndarray:
    S, T, N = J_obs.shape
    J_max = np.ones((S, T), dtype=float)
    for s in range(S):
        for t in range(T):
            mask = active_mask[s, t]
            vals = J_obs[s, t, mask]
            vals = vals[vals > 0]
            if vals.size == 0:
                continue
            if policy == "p95_obs":
                J_max[s, t] = float(np.percentile(vals, 95))
            else:
                J_max[s, t] = float(np.max(vals))
    return J_max


def _std_with_lambda(tilde: np.ndarray, J_max: float, lam: float) -> float:
    J = J_max * sigmoid(lam * tilde)
    return float(np.std(J))


def match_lambda_for_std(tildeJ: np.ndarray, J_max: np.ndarray, target_std: np.ndarray, active_mask: np.ndarray,
                          lam_min: float = 0.1, lam_max: float = 6.0, iters: int = 30) -> np.ndarray:
    S, T, N = tildeJ.shape
    lambda_st = np.ones((S, T), dtype=float)
    for s in range(S):
        for t in range(T):
            mask = active_mask[s, t]
            vals = tildeJ[s, t, mask]
            if vals.size == 0:
                continue
            tgt = float(target_std[s, t])
            if not np.isfinite(tgt) or tgt <= 0:
                lambda_st[s, t] = 1.0
                continue
            lo, hi = lam_min, lam_max
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                cur = _std_with_lambda(vals, J_max[s, t], mid)
                if cur < tgt:
                    lo = mid
                else:
                    hi = mid
            lambda_st[s, t] = 0.5 * (lo + hi)
    return lambda_st
