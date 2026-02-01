from __future__ import annotations

import numpy as np


def estimate_rw_sigma(J_obs: np.ndarray, active_mask: np.ndarray) -> float:
    # simple heuristic: std of week-to-week differences on active set
    S, T, N = J_obs.shape
    diffs = []
    for s in range(S):
        for i in range(N):
            series = J_obs[s, :, i]
            mask = active_mask[s, :, i]
            vals = series[mask]
            if vals.size < 2:
                continue
            diffs.extend(np.diff(vals).tolist())
    if not diffs:
        return 1.0
    return float(np.std(diffs))
