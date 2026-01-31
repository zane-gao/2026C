"""
PyTorch versions of rule-based likelihood computations.
"""
from __future__ import annotations

import torch
import numpy as np
from typing import List

from ..types import RuleType, RuleParams, WeekObs
from .torch_utils import masked_softmax, soft_rank, sigmoid


def loglik_percent_fast(theta: torch.Tensor, obs: WeekObs, params: RuleParams) -> torch.Tensor:
    """Fast percent rule likelihood in PyTorch."""
    if not obs.valid:
        return torch.tensor(0.0, device=theta.device, dtype=theta.dtype)
    
    active = torch.tensor(obs.active_mask, device=theta.device, dtype=theta.dtype)
    if active.sum() <= 0:
        return torch.tensor(0.0, device=theta.device, dtype=theta.dtype)

    elim_idx = np.where(obs.elim_mask > 0)[0].tolist()
    if len(elim_idx) == 0:
        return torch.tensor(0.0, device=theta.device, dtype=theta.dtype)

    J_pct = torch.tensor(obs.J_pct, device=theta.device, dtype=theta.dtype)
    
    # simplified: single-step softmin on combined score
    F = masked_softmax(theta, active)
    C = J_pct + F
    scores = -params.kappa * C + torch.log(active + 1e-12)
    p = masked_softmax(scores, active)
    return torch.log(p[elim_idx].sum() + 1e-12)


def loglik_rank_fast(theta: torch.Tensor, obs: WeekObs, params: RuleParams) -> torch.Tensor:
    """Fast rank rule likelihood in PyTorch."""
    if not obs.valid:
        return torch.tensor(0.0, device=theta.device, dtype=theta.dtype)
    
    active = torch.tensor(obs.active_mask, device=theta.device, dtype=theta.dtype)
    if active.sum() <= 0:
        return torch.tensor(0.0, device=theta.device, dtype=theta.dtype)

    elim_idx = np.where(obs.elim_mask > 0)[0].tolist()
    if len(elim_idx) == 0:
        return torch.tensor(0.0, device=theta.device, dtype=theta.dtype)

    J_z = torch.tensor(obs.J_z, device=theta.device, dtype=theta.dtype)
    
    # simplified: use raw scores (theta + J_z) as "goodness"
    goodness = theta + J_z
    badness = -goodness
    scores = params.kappa_r * badness + torch.log(active + 1e-12)
    p = masked_softmax(scores, active)
    return torch.log(p[elim_idx].sum() + 1e-12)


def loglik_bottom2_fast(theta: torch.Tensor, obs: WeekObs, params: RuleParams) -> torch.Tensor:
    """Fast bottom2_save rule likelihood in PyTorch."""
    if not obs.valid:
        return torch.tensor(0.0, device=theta.device, dtype=theta.dtype)
    
    active = torch.tensor(obs.active_mask, device=theta.device, dtype=theta.dtype)
    if active.sum() <= 1:
        return torch.tensor(0.0, device=theta.device, dtype=theta.dtype)

    elim_idx = np.where(obs.elim_mask > 0)[0].tolist()
    if len(elim_idx) == 0:
        return torch.tensor(0.0, device=theta.device, dtype=theta.dtype)

    J_z = torch.tensor(obs.J_z, device=theta.device, dtype=theta.dtype)
    
    # simplified: use badness scores
    goodness = theta + J_z
    badness = -goodness
    scores = params.kappa_b * badness + torch.log(active + 1e-12)
    p = masked_softmax(scores, active)
    return torch.log(p[elim_idx].sum() + 1e-12)


def loglik_torch(theta: torch.Tensor, obs: WeekObs, params: RuleParams) -> torch.Tensor:
    """Dispatch to appropriate likelihood function based on rule type."""
    if obs.rule == RuleType.PERCENT:
        return loglik_percent_fast(theta, obs, params)
    elif obs.rule == RuleType.RANK:
        return loglik_rank_fast(theta, obs, params)
    elif obs.rule == RuleType.BOTTOM2_SAVE:
        return loglik_bottom2_fast(theta, obs, params)
    else:
        return torch.tensor(0.0, device=theta.device, dtype=theta.dtype)


def compute_total_loglik(
    theta: torch.Tensor,  # [S, T, N]
    week_obs: List[WeekObs],
    season_idx: dict,
    params: RuleParams
) -> torch.Tensor:
    """Compute total log-likelihood across all observations."""
    total_logp = torch.tensor(0.0, device=theta.device, dtype=theta.dtype)
    
    for obs in week_obs:
        s_idx = season_idx[obs.season]
        t_idx = obs.week - 1
        theta_w = theta[s_idx, t_idx, :]
        total_logp = total_logp + loglik_torch(theta_w, obs, params)
    
    return total_logp
