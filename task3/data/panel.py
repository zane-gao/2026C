from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import numpy as np

from ..io.dataset import Dataset, build_masks, load_wide_rows
from ..io.social_proxy import load_social_proxy_map, normalize_name


def _parse_float(val) -> float:
    try:
        if val is None or val == "":
            return float("nan")
        return float(val)
    except Exception:
        return float("nan")


def _parse_int(val) -> int:
    try:
        return int(str(val).strip())
    except Exception:
        return 0


def _build_season_rows(data_csv: str) -> Dict[int, List[Dict[str, str]]]:
    rows = load_wide_rows(data_csv)
    season_map: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        season = _parse_int(row.get("season"))
        if season <= 0:
            continue
        season_map.setdefault(season, []).append(row)
    return season_map


def compute_ref_couples(theta: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    if theta.ndim != 4:
        raise ValueError("theta must be [K,S,T,N]")
    K, S, T, N = theta.shape
    ref_idx = np.full((S, T), -1, dtype=int)
    for s in range(S):
        for t in range(T):
            mask = active_mask[s, t]
            if mask.sum() == 0:
                continue
            f_mean = np.zeros((N,), dtype=float)
            for k in range(K):
                theta_st = theta[k, s, t]
                maxv = np.max(theta_st[mask])
                expv = np.exp(theta_st - maxv)
                expv[~mask] = 0.0
                denom = expv[mask].sum()
                if denom <= 0:
                    continue
                f_mean += expv / denom
            f_mean /= max(K, 1)
            ref = int(np.argmax(f_mean))
            if not mask[ref]:
                ref = int(np.where(mask)[0][0])
            ref_idx[s, t] = ref
    return ref_idx


def build_panel(
    dataset: Dataset,
    data_csv: str,
    social_path: Optional[str] = None,
    active_mask: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    elim_mask: Optional[np.ndarray] = None,
    withdraw_mask: Optional[np.ndarray] = None,
    ref_idx: Optional[np.ndarray] = None,
) -> "object":
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required to build panel") from exc

    if active_mask is None or valid_mask is None or elim_mask is None or withdraw_mask is None:
        masks = build_masks(dataset)
        active_mask = masks["active_mask"]
        valid_mask = masks["valid_mask"]
        elim_mask = masks["elim_mask"]
        withdraw_mask = masks["withdraw_mask"]

    season_map = _build_season_rows(data_csv)
    social_map = load_social_proxy_map(social_path) if social_path else {}

    S, T, N = dataset.J_obs.shape
    # compute per-week stats within active set
    mean_st = np.full((S, T), np.nan, dtype=float)
    std_st = np.full((S, T), np.nan, dtype=float)
    sum_st = np.full((S, T), np.nan, dtype=float)
    for s in range(S):
        for t in range(T):
            mask = active_mask[s, t]
            if mask.sum() == 0:
                continue
            vals = dataset.J_obs[s, t, mask]
            mean_st[s, t] = float(np.mean(vals))
            std = float(np.std(vals))
            std_st[s, t] = std if std > 0 else float("nan")
            sum_st[s, t] = float(np.sum(vals))

    celeb_id_map: Dict[str, int] = {}
    pro_id_map: Dict[str, int] = {}
    couple_id_map: Dict[Tuple[int, str, str], int] = {}

    rows_out: List[Dict[str, object]] = []
    for s_idx, season in enumerate(dataset.seasons):
        rows_s = season_map.get(season, [])
        for i_idx in range(min(len(rows_s), N)):
            row = rows_s[i_idx]
            celeb_name = str(row.get("celebrity_name", "")).strip()
            pro_name = str(row.get("ballroom_partner", "")).strip()
            celeb_key = normalize_name(celeb_name)
            pro_key = normalize_name(pro_name)
            if celeb_key not in celeb_id_map:
                celeb_id_map[celeb_key] = len(celeb_id_map)
            if pro_key not in pro_id_map:
                pro_id_map[pro_key] = len(pro_id_map)
            couple_key = (season, celeb_key, pro_key)
            if couple_key not in couple_id_map:
                couple_id_map[couple_key] = len(couple_id_map)

            industry = str(row.get("celebrity_industry", "")).strip()
            homestate = str(row.get("celebrity_homestate", "")).strip()
            homecountry = str(row.get("celebrity_homecountry/region", "")).strip()
            age = _parse_float(row.get("celebrity_age_during_season"))

            social_key = (season, celeb_key, pro_key)
            srec = social_map.get(social_key, {}) if social_map else {}
            p_cele = srec.get("P_cele")
            p_partner = srec.get("P_partner")
            missing_cele = bool(srec.get("missing_cele_total", True))
            missing_partner = bool(srec.get("missing_partner_total", True))

            extra_social = {}
            for key, val in srec.items():
                if key.startswith("P_cele_") or key.startswith("P_partner_"):
                    extra_social[key] = val
                if key.startswith("missing_cele_") or key.startswith("missing_partner_"):
                    extra_social[key] = val

            for t_idx in range(T):
                if not dataset.has_any_score[s_idx, t_idx, i_idx]:
                    continue
                j_raw = float(dataset.J_obs[s_idx, t_idx, i_idx])
                j_mean = mean_st[s_idx, t_idx]
                j_std = std_st[s_idx, t_idx]
                j_sum = sum_st[s_idx, t_idx]
                is_active = bool(active_mask[s_idx, t_idx, i_idx])

                j_z = float("nan")
                j_pct = float("nan")
                if is_active and math.isfinite(j_mean) and math.isfinite(j_std) and j_std > 0:
                    j_z = (j_raw - j_mean) / j_std
                if is_active and math.isfinite(j_sum) and j_sum > 0:
                    j_pct = j_raw / j_sum

                rows_out.append({
                    "season": season,
                    "week": t_idx + 1,
                    "couple_id": couple_id_map[couple_key],
                    "celeb_id": celeb_id_map[celeb_key],
                    "pro_id": pro_id_map[pro_key],
                    "celeb_name": celeb_name,
                    "pro_name": pro_name,
                    "couple_name": f"{celeb_name} / {pro_name}" if celeb_name or pro_name else "",
                    "age": age,
                    "industry": industry,
                    "homestate": homestate,
                    "homecountry": homecountry,
                    "J_raw": j_raw,
                    "J_z": j_z,
                    "J_pct": j_pct,
                    "is_active": is_active,
                    "is_valid_week": bool(valid_mask[s_idx, t_idx]),
                    "eliminated_this_week": bool(elim_mask[s_idx, t_idx, i_idx]),
                    "withdrew_this_week": bool(withdraw_mask[s_idx, t_idx, i_idx]),
                    "P_cele": float(p_cele) if p_cele is not None else float("nan"),
                    "P_partner": float(p_partner) if p_partner is not None else float("nan"),
                    "missing_cele_total": bool(missing_cele),
                    "missing_partner_total": bool(missing_partner),
                    "s_idx": s_idx,
                    "t_idx": t_idx,
                    "i_idx": i_idx,
                })
                if extra_social:
                    rows_out[-1].update(extra_social)

    panel = pd.DataFrame(rows_out)
    if ref_idx is not None:
        ref_flat = []
        for _, row in panel.iterrows():
            ref = int(ref_idx[int(row["s_idx"]), int(row["t_idx"])])
            ref_flat.append(ref)
        panel["ref_couple_id"] = ref_flat
    return panel
