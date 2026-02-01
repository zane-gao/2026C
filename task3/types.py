from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ModelType(str, Enum):
    M1_JUDGES = "m1_judges"
    M2_FANS = "m2_fans"
    M3_SURVIVAL = "m3_survival"
    JOINT = "joint"


@dataclass
class FeatureSpec:
    base_numeric: List[str] = field(default_factory=lambda: ["age"])
    base_categorical: List[str] = field(default_factory=lambda: ["industry", "homestate", "homecountry"])
    social_cols: List[str] = field(default_factory=lambda: ["P_cele", "P_partner"])
    missing_cols: List[str] = field(default_factory=lambda: ["missing_cele_total", "missing_partner_total"])
    platform_cols: List[str] = field(default_factory=list)
    include_intercept: bool = True


@dataclass
class PanelMeta:
    seasons: List[int]
    k_index: List[int]
    ref_couple_id: Optional[List[List[int]]] = None
