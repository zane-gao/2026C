from __future__ import annotations

import numpy as np


def fit_linear_skill(J_obs: np.ndarray, active_mask: np.ndarray, valid_mask: np.ndarray | None = None):
    S, T, N = J_obs.shape
    a = np.zeros((S, N), dtype=float)
    b = np.zeros((S, N), dtype=float)
    alpha = np.zeros((S, T), dtype=float)
    residuals: list[float] = []

    for s in range(S):
        rows = []
        ys = []
        for t in range(T):
            if valid_mask is not None and not bool(valid_mask[s, t]):
                continue
            for i in range(N):
                if not active_mask[s, t, i]:
                    continue
                y = J_obs[s, t, i]
                if not np.isfinite(y) or y <= 0:
                    continue
                # design vector: a_i, b_i, alpha_t (t>=2)
                x = np.zeros((2 * N + (T - 1),), dtype=float)
                x[i] = 1.0
                x[N + i] = float(t + 1)
                if t >= 1:
                    x[2 * N + (t - 1)] = 1.0
                rows.append(x)
                ys.append(float(y))
        if not rows:
            continue
        X = np.vstack(rows)
        y = np.array(ys, dtype=float)
        # ridge for stability
        ridge = 1e-6
        XtX = X.T @ X + ridge * np.eye(X.shape[1])
        Xty = X.T @ y
        coef = np.linalg.solve(XtX, Xty)
        a[s] = coef[:N]
        b[s] = coef[N:2 * N]
        alpha[s, 1:] = coef[2 * N:]
        # residuals
        y_hat = X @ coef
        residuals.extend((y - y_hat).tolist())

    sigma_J = float(np.std(residuals)) if residuals else 1.0
    return {"a": a, "b": b, "alpha": alpha, "sigma_J": sigma_J}
