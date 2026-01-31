from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import numpy as np


class RuleType(str, Enum):
    RANK = "rank"
    PERCENT = "percent"
    BOTTOM2_SAVE = "bottom2_save"


@dataclass(frozen=True)
class RuleParams:
    epsilon: float
    delta: float
    tau: float
    kappa: float
    kappa_r: float
    alpha: float
    kappa_b: float
    eta1: float
    eta2: float
    lambda_ent: float
    bottom2_base: str = "rank"
    loglik_mode: str = "full"


@dataclass(frozen=True)
class WeekObs:
    season: int
    week: int
    rule: RuleType
    valid: bool
    active_mask: np.ndarray  # shape [N]
    elim_mask: np.ndarray    # shape [N]
    J: np.ndarray            # shape [N]
    J_z: np.ndarray          # shape [N]
    J_pct: np.ndarray        # shape [N]
    J_cum_avg: np.ndarray    # shape [N]


@dataclass(frozen=True)
class TensorPack:
    seasons: List[int]
    T_max: int
    N_max: int
    X: np.ndarray                # [S, N, D]
    J: np.ndarray                # [S, T, N]
    J_z: np.ndarray              # [S, T, N]
    J_pct: np.ndarray            # [S, T, N]
    J_cum_avg: np.ndarray        # [S, T, N]
    active_mask: np.ndarray      # [S, T, N]
    elim_mask: np.ndarray        # [S, T, N]
    valid_mask: np.ndarray       # [S, T]
    rule_id: np.ndarray          # [S, T]
    pro_id: np.ndarray           # [S, N]
    celeb_id: np.ndarray         # [S, N]
    couple_names: List[List[str]]  # per season, list of couple display names
    feature_names: List[str]
