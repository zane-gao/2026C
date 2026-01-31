from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PriorConfig:
    beta_sigma: float = 1.0
    u_sigma: float = 1.0
    v_sigma: float = 1.0
    w_sigma: float = 1.0
    gamma_sigma: float = 1.0
    sigma_rw: float = 0.5
    sigma_delta0: float = 0.5
