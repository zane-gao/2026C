from __future__ import annotations

import numpy as np

from .utils import masked_softmax, soft_rank, sigmoid
from ..types import RuleParams, WeekObs


def _compute_badness_np(theta, obs: WeekObs, params: RuleParams):
    active = obs.active_mask.astype(float)
    if params.bottom2_base == "percent":
        F = masked_softmax(theta, active)
        C = obs.J_pct + F
        return C
    # rank base
    rJ = soft_rank(obs.J, params.tau)
    rF = soft_rank(theta, params.tau)
    rC = rJ + rF + params.alpha * rF
    return rC


def loglik_np(theta, obs: WeekObs, params: RuleParams) -> float:
    if not obs.valid:
        return 0.0
    active = obs.active_mask.astype(float)
    if active.sum() <= 1:
        return 0.0

    elim_idx = np.where(obs.elim_mask > 0)[0].tolist()
    if len(elim_idx) == 0:
        return 0.0

    badness = _compute_badness_np(theta, obs, params)
    if len(elim_idx) != 1:
        # fallback: multi-elim -> sequential soft selection on badness
        mask = active.copy()
        logp = 0.0
        for e in elim_idx:
            scores = params.kappa_b * badness + np.log(mask + 1e-12)
            p = masked_softmax(scores, mask)
            logp += float(np.log(p[e] + 1e-12))
            mask[e] = 0.0
        return logp

    elim = elim_idx[0]
    p_b1 = masked_softmax(params.kappa_b * badness, active)

    total_prob = 0.0
    idxs = np.where(active > 0)[0]
    for i in idxs:
        # second draw
        mask2 = active.copy()
        mask2[i] = 0.0
        p_b2 = masked_softmax(params.kappa_b * badness, mask2)
        for j in idxs:
            if j == i:
                continue
            pair_prob = p_b1[i] * p_b2[j]
            if pair_prob <= 0:
                continue
            # elimination probability within {i,j}
            score = params.eta1 * (obs.J[j] - obs.J[i]) + params.eta2 * (obs.J_cum_avg[j] - obs.J_cum_avg[i])
            p_elim_i = sigmoid(score)
            if elim == i:
                p_elim = p_elim_i
            elif elim == j:
                p_elim = 1.0 - p_elim_i
            else:
                p_elim = 0.0
            total_prob += pair_prob * p_elim

    return float(np.log(total_prob + 1e-12))


# pytensor version
try:
    import pytensor.tensor as pt  # type: ignore
    from .pt_utils import masked_softmax as pt_softmax, soft_rank as pt_soft_rank, sigmoid as pt_sigmoid
except Exception:
    pt = None


def _compute_badness_pt(theta, obs: WeekObs, params: RuleParams):
    active = obs.active_mask.astype(float)
    if params.bottom2_base == "percent":
        F = pt_softmax(theta, active)
        C = obs.J_pct + F
        return C
    rJ = pt_soft_rank(obs.J, params.tau)
    rF = pt_soft_rank(theta, params.tau)
    rC = rJ + rF + params.alpha * rF
    return rC


def loglik_pt(theta, obs: WeekObs, params: RuleParams):
    if not obs.valid or pt is None:
        return 0.0
    active = obs.active_mask.astype(float)
    if active.sum() <= 1:
        return 0.0

    elim_idx = np.where(obs.elim_mask > 0)[0].tolist()
    if len(elim_idx) == 0:
        return 0.0

    badness = _compute_badness_pt(theta, obs, params)
    if len(elim_idx) != 1:
        # fallback: multi-elim -> sequential soft selection on badness
        mask = active.copy()
        logp = 0.0
        for e in elim_idx:
            scores = params.kappa_b * badness + pt.log(mask + 1e-12)
            p = pt_softmax(scores, mask)
            logp = logp + pt.log(p[e] + 1e-12)
            one_hot = np.zeros_like(active)
            one_hot[e] = 1.0
            mask = mask * (1.0 - one_hot)
        return logp

    elim = elim_idx[0]
    p_b1 = pt_softmax(params.kappa_b * badness, active)

    total_prob = 0.0
    idxs = np.where(active > 0)[0]
    for i in idxs:
        mask2 = active.copy()
        mask2[i] = 0.0
        p_b2 = pt_softmax(params.kappa_b * badness, mask2)
        for j in idxs:
            if j == i:
                continue
            pair_prob = p_b1[i] * p_b2[j]
            score = params.eta1 * (obs.J[j] - obs.J[i]) + params.eta2 * (obs.J_cum_avg[j] - obs.J_cum_avg[i])
            p_elim_i = pt_sigmoid(score)
            if elim == i:
                p_elim = p_elim_i
            elif elim == j:
                p_elim = 1.0 - p_elim_i
            else:
                p_elim = 0.0
            total_prob = total_prob + pair_prob * p_elim

    return pt.log(total_prob + 1e-12)


def loglik_pt_fast(theta, obs: WeekObs, params: RuleParams):
    if not obs.valid or pt is None:
        return 0.0
    active = obs.active_mask.astype(float)
    if active.sum() <= 1:
        return 0.0

    elim_idx = np.where(obs.elim_mask > 0)[0].tolist()
    if len(elim_idx) == 0:
        return 0.0

    # simplified: single-step bottom selection on badness, ignore save dynamics
    if params.bottom2_base == "percent":
        F = pt_softmax(theta, active)
        badness = -(obs.J_pct + F)
    else:
        goodness = theta + obs.J_z
        badness = -goodness

    scores = params.kappa_b * badness + pt.log(active + 1e-12)
    p = pt_softmax(scores, active)
    return pt.log(pt.sum(p[elim_idx]) + 1e-12)
