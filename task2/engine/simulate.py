from __future__ import annotations

import numpy as np

from ..data.calibrate import sigmoid
from ..data.indexing import compute_final_rank, compute_survival_time
from ..preference.mode_a import compute_F_mode_a
from ..preference.mode_b import compute_F_mode_b
from ..mechanisms.percent import compute_percent_scores, select_elim_percent, margin_percent
from ..mechanisms.rank import compute_rank_scores, select_elim_rank, margin_rank
from ..mechanisms.judges_save import select_bottom2, decide_save
from ..mechanisms.mixture import soft_probs_percent, soft_probs_rank, mix_probs, select_elim_mix


def _compute_J(tildeJ: np.ndarray, J_max: np.ndarray, lambda_st: np.ndarray) -> np.ndarray:
    return J_max[None, :, :, None] * sigmoid(lambda_st[None, :, :, None] * tildeJ)


def _compute_J_z(J_t: np.ndarray, active: np.ndarray) -> np.ndarray:
    J_z = np.zeros_like(J_t)
    idx = np.where(active)[0]
    if idx.size == 0:
        return J_z
    vals = J_t[idx]
    mean = vals.mean()
    std = vals.std()
    if std < 1e-6:
        return J_z
    J_z[idx] = (vals - mean) / std
    return J_z


def simulate_trajectories(mode: str, mechanism: str, theta_samples: np.ndarray,
                          mu: np.ndarray | None, gamma: np.ndarray | None, epsilon: np.ndarray | None,
                          S_samples: np.ndarray, tildeJ: np.ndarray,
                          valid_mask: np.ndarray, active_init: np.ndarray, withdraw_mask: np.ndarray,
                          multi_elim_count: np.ndarray, J_max: np.ndarray, lambda_st: np.ndarray,
                          tie_mode: str, soft_percent: bool, soft_rank: bool,
                          crn: dict, js_config: dict, mix_config: dict) -> dict:
    K, S, T, N = S_samples.shape
    J = _compute_J(tildeJ, J_max, lambda_st)

    winner = -np.ones((K, S), dtype=int)
    final_rank = np.zeros((K, S, N), dtype=float)
    elim_matrix = np.zeros((K, S, T, N), dtype=bool)
    active_mask = np.zeros((K, S, T, N), dtype=bool)
    survival_time = np.zeros((K, S, N), dtype=int)
    F_bar = np.zeros((K, S, N), dtype=float)
    margin = np.full((K, S, T), np.nan, dtype=float)

    bottom2_mask = None
    saved_mask = None
    if mechanism == "JS":
        bottom2_mask = np.zeros((K, S, T, N), dtype=bool)
        saved_mask = np.zeros((K, S, T, N), dtype=bool)

    # precompute S_bar (season mean over valid weeks)
    valid_counts = np.maximum(valid_mask.sum(axis=1), 1)
    S_bar = (S_samples * valid_mask[None, :, :, None]).sum(axis=2) / valid_counts[None, :, None]

    for k in range(K):
        for s in range(S):
            active = active_init[s].copy()
            meanJ_hist = np.zeros((N,), dtype=float)
            meanJ_count = np.zeros((N,), dtype=float)
            for t in range(T):
                active_mask[k, s, t] = active
                if active.sum() == 0:
                    continue
                J_t = J[k, s, t]
                J_z = _compute_J_z(J_t, active)

                # update meanJ history for JS
                idx_active = np.where(active)[0]
                meanJ_count[idx_active] += 1.0
                meanJ_hist[idx_active] += J_t[idx_active]
                meanJ = np.zeros_like(meanJ_hist)
                meanJ[meanJ_count > 0] = meanJ_hist[meanJ_count > 0] / meanJ_count[meanJ_count > 0]

                if mode == "A":
                    theta_t = theta_samples[k, s, t]
                    F_t = compute_F_mode_a(theta_t, active)
                else:
                    if mu is None or gamma is None or epsilon is None:
                        theta_t = theta_samples[k, s, t]
                        F_t = compute_F_mode_a(theta_t, active)
                    else:
                        mu_s = mu[k, s] if mu.ndim == 3 else mu[s]
                        gamma_k = gamma[k] if np.ndim(gamma) > 0 else float(gamma)
                        eps_t = epsilon[k, s, t]
                        F_t = compute_F_mode_b(mu_s, gamma_k, eps_t, J_z, active)

                if valid_mask[s, t]:
                    F_bar[k, s] += F_t

                d = int(multi_elim_count[s, t])
                # Ensure we continue eliminating until only 1 remains
                # If d=0 but we still have multiple active, we're in "extra" finals
                n_active = int(active.sum())
                if d <= 0 and n_active > 1:
                    # Finals: eliminate 1 per "extra week" until winner is decided
                    d = 1
                if d <= 0 or n_active <= d:
                    # still apply withdrawal
                    active = active & (~withdraw_mask[s, t])
                    continue

                tie_noise = crn["tie_noise"][k, s, t]
                gumbel_noise = crn["gumbel_noise"][k, s, t]

                if mechanism == "P":
                    C = compute_percent_scores(J_t, F_t, active)
                    margin[k, s, t] = margin_percent(C, active, d)
                    elim_idx = select_elim_percent(C, active, d, tie_noise=tie_noise,
                                                   soft=soft_percent, gumbel_noise=gumbel_noise,
                                                   kappa=mix_config.get("kappa", 20.0))
                elif mechanism == "R":
                    R = compute_rank_scores(J_t, F_t, active, tie_mode=tie_mode, tie_noise=tie_noise)
                    margin[k, s, t] = margin_rank(R, active, d)
                    elim_idx = select_elim_rank(R, active, d, tie_noise=tie_noise,
                                                soft=soft_rank, gumbel_noise=gumbel_noise,
                                                kappa_r=mix_config.get("kappa_r", 10.0))
                elif mechanism == "JS":
                    # bottom2 base
                    base = js_config.get("bottom2_base", "rank")
                    if base == "percent":
                        C = compute_percent_scores(J_t, F_t, active)
                        base_scores = C
                        margin[k, s, t] = margin_percent(C, active, d)
                    else:
                        R = compute_rank_scores(J_t, F_t, active, tie_mode=tie_mode, tie_noise=tie_noise)
                        base_scores = -R
                        margin[k, s, t] = margin_rank(R, active, d)
                    if d == 1:
                        bottom2 = select_bottom2(base_scores, active, tie_noise=tie_noise)
                        if bottom2.size == 2:
                            bottom2_mask[k, s, t, bottom2] = True
                        u = float(crn["save_u"][k, s, t])
                        elim_one, save_one = decide_save(J_t, meanJ, bottom2,
                                                         js_config.get("eta1", 1.0),
                                                         js_config.get("eta2", 0.5),
                                                         js_config.get("deterministic", False),
                                                         u)
                        elim_idx = np.array([elim_one], dtype=int) if elim_one >= 0 else np.array([], dtype=int)
                        if save_one >= 0:
                            saved_mask[k, s, t, save_one] = True
                    else:
                        elim_idx = select_elim_rank(base_scores, active, d, tie_noise=tie_noise,
                                                    soft=soft_rank, gumbel_noise=gumbel_noise,
                                                    kappa_r=mix_config.get("kappa_r", 10.0))
                else:
                    # mixture mechanism
                    C = compute_percent_scores(J_t, F_t, active)
                    R = compute_rank_scores(J_t, F_t, active, tie_mode=tie_mode, tie_noise=tie_noise)
                    pP = soft_probs_percent(C, active, mix_config.get("kappa", 20.0))
                    pR = soft_probs_rank(R, active, mix_config.get("kappa_r", 10.0))
                    w = float(mix_config.get("w", 0.5))
                    p = mix_probs(pP, pR, w)
                    elim_idx = select_elim_mix(p, active, d, gumbel_noise=gumbel_noise)
                    margin[k, s, t] = margin_percent(C, active, d)

                if elim_idx.size > 0:
                    elim_matrix[k, s, t, elim_idx] = True
                    active[elim_idx] = False
                # apply withdrawal
                active = active & (~withdraw_mask[s, t])

            # final ranks
            final_rank[k, s] = compute_final_rank(elim_matrix[k, s], active_init[s])
            survival_time[k, s] = compute_survival_time(active_mask[k, s])
            # winner
            if active_init[s].any():
                winner_idx = np.where(final_rank[k, s] == 1)[0]
                if winner_idx.size > 0:
                    winner[k, s] = int(winner_idx[0])

    # average F over valid weeks
    valid_counts = np.maximum(valid_mask.sum(axis=1), 1)
    F_bar = F_bar / valid_counts[None, :, None]

    return {
        "winner": winner,
        "final_rank": final_rank,
        "elim_matrix": elim_matrix,
        "active_mask": active_mask,
        "survival_time": survival_time,
        "F_bar": F_bar,
        "S_bar": S_bar,
        "margin": margin,
        "bottom2_mask": bottom2_mask,
        "saved_mask": saved_mask,
    }
