from __future__ import annotations

import numpy as np


def compute_ite(traj_r: dict, traj_p: dict) -> dict:
    T_r = traj_r["survival_time"]
    T_p = traj_p["survival_time"]
    rank_r = traj_r["final_rank"]
    rank_p = traj_p["final_rank"]

    tau_T = T_r - T_p
    tau_pi = rank_r - rank_p

    ite = {
        "tau_T_mean": tau_T.mean(axis=0),
        "tau_T_q05": np.quantile(tau_T, 0.05, axis=0),
        "tau_T_q95": np.quantile(tau_T, 0.95, axis=0),
        "tau_pi_mean": tau_pi.mean(axis=0),
        "tau_pi_q05": np.quantile(tau_pi, 0.05, axis=0),
        "tau_pi_q95": np.quantile(tau_pi, 0.95, axis=0),
    }
    return ite


def compute_group_ate(ite: dict, group_mask: np.ndarray) -> dict:
    # group_mask: [S, N]
    tau_T = ite["tau_T_mean"]
    tau_pi = ite["tau_pi_mean"]
    group = group_mask.astype(bool)
    vals_T = tau_T[group]
    vals_pi = tau_pi[group]
    return {
        "ATE_T": float(np.mean(vals_T)) if vals_T.size else float('nan'),
        "ATE_pi": float(np.mean(vals_pi)) if vals_pi.size else float('nan'),
    }
