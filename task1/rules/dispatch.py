from __future__ import annotations

from ..types import RuleType, RuleParams, WeekObs
from . import percent, rank, bottom2_save


def loglik_np(theta, obs: WeekObs, params: RuleParams) -> float:
    if obs.rule == RuleType.PERCENT:
        return percent.loglik_np(theta, obs, params)
    if obs.rule == RuleType.RANK:
        return rank.loglik_np(theta, obs, params)
    if obs.rule == RuleType.BOTTOM2_SAVE:
        return bottom2_save.loglik_np(theta, obs, params)
    return 0.0


def loglik_pt(theta, obs: WeekObs, params: RuleParams):
    if getattr(params, "loglik_mode", "full") == "fast":
        if obs.rule == RuleType.PERCENT:
            return percent.loglik_pt_fast(theta, obs, params)
        if obs.rule == RuleType.RANK:
            return rank.loglik_pt_fast(theta, obs, params)
        if obs.rule == RuleType.BOTTOM2_SAVE:
            return bottom2_save.loglik_pt_fast(theta, obs, params)
    if obs.rule == RuleType.PERCENT:
        return percent.loglik_pt(theta, obs, params)
    if obs.rule == RuleType.RANK:
        return rank.loglik_pt(theta, obs, params)
    if obs.rule == RuleType.BOTTOM2_SAVE:
        return bottom2_save.loglik_pt(theta, obs, params)
    return 0.0
