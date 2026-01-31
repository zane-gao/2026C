from __future__ import annotations

import numpy as np

from .utils import masked_softmax, sigmoid
from ..types import RuleParams, WeekObs


def _sequential_softmin_logp(C, elim_idx, active_mask, kappa):
    mask = active_mask.astype(float).copy()
    logp = 0.0
    for e in elim_idx:
        scores = -kappa * C + np.log(mask + 1e-12)
        p = masked_softmax(scores, mask)
        logp += float(np.log(p[e] + 1e-12))
        mask[e] = 0.0
    return logp


def loglik_np(theta, obs: WeekObs, params: RuleParams) -> float:
    if not obs.valid:
        return 0.0
    active = obs.active_mask.astype(float)
    if active.sum() <= 0:
        return 0.0

    elim_idx = np.where(obs.elim_mask > 0)[0].tolist()
    if len(elim_idx) == 0:
        return 0.0

    F = masked_softmax(theta, active)
    C = obs.J_pct + F

    # soft constraint
    survivors = np.where((obs.elim_mask == 0) & (active > 0))[0]
    log_con = 0.0
    for e in elim_idx:
        for j in survivors:
            log_con += np.log(sigmoid((C[j] - C[e] - params.epsilon) / params.delta) + 1e-12)

    logp = _sequential_softmin_logp(C, elim_idx, active, params.kappa)
    return logp + log_con


# pytensor version
try:
    import pytensor.tensor as pt  # type: ignore
    from .pt_utils import masked_softmax as pt_softmax, sigmoid as pt_sigmoid
except Exception:
    pt = None


def _sequential_softmin_logp_pt(C, elim_idx, active_mask, kappa):
    mask = pt.cast(active_mask, "float32")
    logp = 0.0
    for e in elim_idx:
        scores = -kappa * C + pt.log(mask + 1e-12)
        p = pt_softmax(scores, mask)
        logp = logp + pt.log(p[e] + 1e-12)
        # zero-out selected
        one_hot = np.zeros_like(active_mask)
        one_hot[e] = 1.0
        mask = mask * (1.0 - one_hot)
    return logp


def loglik_pt(theta, obs: WeekObs, params: RuleParams):
    if not obs.valid or pt is None:
        return 0.0
    active = obs.active_mask.astype(float)
    if active.sum() <= 0:
        return 0.0

    elim_idx = np.where(obs.elim_mask > 0)[0].tolist()
    if len(elim_idx) == 0:
        return 0.0

    F = pt_softmax(theta, active)
    C = obs.J_pct + F

    survivors = np.where((obs.elim_mask == 0) & (active > 0))[0]
    log_con = 0.0
    for e in elim_idx:
        for j in survivors:
            log_con = log_con + pt.log(pt_sigmoid((C[j] - C[e] - params.epsilon) / params.delta) + 1e-12)

    logp = _sequential_softmin_logp_pt(C, elim_idx, active, params.kappa)
    return logp + log_con


def loglik_pt_fast(theta, obs: WeekObs, params: RuleParams):
    if not obs.valid or pt is None:
        return 0.0
    active = obs.active_mask.astype(float)
    if active.sum() <= 0:
        return 0.0

    elim_idx = np.where(obs.elim_mask > 0)[0].tolist()
    if len(elim_idx) == 0:
        return 0.0

    # simplified: single-step softmin on combined score, no constraint loops
    F = pt_softmax(theta, active)
    C = obs.J_pct + F
    scores = -params.kappa * C + pt.log(active + 1e-12)
    p = pt_softmax(scores, active)
    return pt.log(pt.sum(p[elim_idx]) + 1e-12)
