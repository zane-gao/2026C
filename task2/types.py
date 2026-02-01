from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import numpy as np


class Mode(str, Enum):
    A = "A"
    B = "B"


class Mechanism(str, Enum):
    PERCENT = "P"
    RANK = "R"
    JUDGES_SAVE = "JS"
    MIXTURE = "W_MIX"


@dataclass
class ArtifactBundle:
    theta: np.ndarray  # [K, S, T, N]
    valid_mask: np.ndarray  # [S, T]
    active_init: np.ndarray  # [S, N]
    withdraw_mask: np.ndarray  # [S, T, N]
    multi_elim_count: np.ndarray  # [S, T]
    season_ids: List[int]
    couple_names: List[List[str]]
    elim_real: Optional[np.ndarray] = None  # [S, T, N]
    mu: Optional[np.ndarray] = None  # [K, S, N] or [S, N]
    gamma: Optional[np.ndarray] = None  # [K] or scalar
    epsilon: Optional[np.ndarray] = None  # [K, S, T, N]
    rule_id: Optional[np.ndarray] = None


@dataclass
class SkillParams:
    a: np.ndarray  # [S, N]
    b: np.ndarray  # [S, N]
    alpha: np.ndarray  # [S, T]
    sigma_J: float
    J_max: np.ndarray  # [S, T]
    lambda_st: np.ndarray  # [S, T]


@dataclass
class TrajectoryPack:
    winner: np.ndarray  # [K, S]
    final_rank: np.ndarray  # [K, S, N]
    elim_matrix: np.ndarray  # [K, S, T, N]
    active_mask: np.ndarray  # [K, S, T, N]
    survival_time: np.ndarray  # [K, S, N]
    F_bar: np.ndarray  # [K, S, N]
    S_bar: np.ndarray  # [K, S, N]
    margin: np.ndarray  # [K, S, T]
    bottom2_mask: Optional[np.ndarray] = None  # [K, S, T, N]
    saved_mask: Optional[np.ndarray] = None  # [K, S, T, N]
