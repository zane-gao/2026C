from __future__ import annotations

import numpy as np

from .utils import masked_softmax, soft_rank
from ..types import RuleParams, WeekObs


def _sequential_softmax_logp(scores, elim_idx, active_mask, kappa):
    mask = active_mask.astype(float).copy()
    logp = 0.0
    for e in elim_idx:
        logits = kappa * scores + np.log(mask + 1e-12)
        p = masked_softmax(logits, mask)
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

    rJ = soft_rank(obs.J, params.tau)
    rF = soft_rank(theta, params.tau)
    rC = rJ + rF + params.alpha * rF

    logp = _sequential_softmax_logp(rC, elim_idx, active, params.kappa_r)
    return logp


# pytensor version
try:
    import pytensor.tensor as pt  # type: ignore
    from .pt_utils import soft_rank as pt_soft_rank
    from .pt_utils import masked_softmax as pt_softmax
except Exception:
    pt = None


def _sequential_softmax_logp_pt(scores, elim_idx, active_mask, kappa):
    mask = pt.cast(active_mask, "float32")
    logp = 0.0
    for e in elim_idx:
        logits = kappa * scores + pt.log(mask + 1e-12)
        p = pt_softmax(logits, mask)
        logp = logp + pt.log(p[e] + 1e-12)
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

    rJ = pt_soft_rank(obs.J, params.tau)
    rF = pt_soft_rank(theta, params.tau)
    rC = rJ + rF + params.alpha * rF

    logp = _sequential_softmax_logp_pt(rC, elim_idx, active, params.kappa_r)
    return logp


def loglik_pt_fast(theta, obs: WeekObs, params: RuleParams):
    if not obs.valid or pt is None:
        return 0.0
    active = obs.active_mask.astype(float)
    if active.sum() <= 0:
        return 0.0

    elim_idx = np.where(obs.elim_mask > 0)[0].tolist()
    if len(elim_idx) == 0:
        return 0.0

    # simplified: use raw scores (theta + J_z) as "goodness"
    goodness = theta + obs.J_z
    badness = -goodness
    scores = params.kappa_r * badness + pt.log(active + 1e-12)
    p = pt_softmax(scores, active)
    return pt.log(pt.sum(p[elim_idx]) + 1e-12)
