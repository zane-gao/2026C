from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


WEEK_SCORE_RE = re.compile(r"^week(\d+)_judge(\d+)_score$")


def _parse_score(val) -> float | None:
    if val is None:
        return None
    v = str(val).strip()
    if v == "" or v.upper() == "N/A":
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def find_data_csv(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    # search by filename
    target = Path(path).name
    root = Path.cwd()
    for dirpath, _, filenames in os.walk(root):
        if target in filenames:
            return str(Path(dirpath) / target)
    raise FileNotFoundError(path)


def _read_text_with_fallback(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def load_wide_rows(path: str) -> List[Dict[str, str]]:
    p = Path(find_data_csv(path))
    text = _read_text_with_fallback(p)
    lines = text.splitlines()
    reader = csv.DictReader(lines)
    return [dict(row) for row in reader]


def detect_week_columns(headers: List[str]) -> Tuple[Dict[Tuple[int, int], str], int]:
    mapping: Dict[Tuple[int, int], str] = {}
    max_week = 0
    for col in headers:
        m = WEEK_SCORE_RE.match(col)
        if not m:
            continue
        week = int(m.group(1))
        judge = int(m.group(2))
        mapping[(week, judge)] = col
        if week > max_week:
            max_week = week
    return mapping, max_week


@dataclass
class Dataset:
    seasons: List[int]
    T_max: int
    N_max: int
    couple_names: List[List[str]]
    J_obs: np.ndarray  # [S, T, N]
    has_any_score: np.ndarray  # [S, T, N]
    results: List[List[str]]
    placement: np.ndarray  # [S, N]



def build_dataset(path: str, max_week: int | None = None) -> Dataset:
    rows = load_wide_rows(path)
    if not rows:
        raise ValueError("No rows loaded from data CSV")

    headers = list(rows[0].keys())
    mapping, detected_max = detect_week_columns(headers)
    if max_week is None:
        max_week = detected_max

    season_map: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        try:
            season = int(str(row.get("season", "0")).strip() or 0)
        except Exception:
            season = 0
        season_map.setdefault(season, []).append(row)

    seasons = sorted([s for s in season_map.keys() if s > 0])
    if not seasons:
        raise ValueError("No valid seasons found in data")

    N_max = max(len(season_map[s]) for s in seasons)
    S = len(seasons)
    T = int(max_week)

    J_obs = np.zeros((S, T, N_max), dtype=float)
    has_any = np.zeros((S, T, N_max), dtype=bool)
    placement = np.full((S, N_max), np.nan, dtype=float)
    couple_names: List[List[str]] = []
    results: List[List[str]] = []

    for s_idx, season in enumerate(seasons):
        rows_s = season_map[season]
        couple_names.append([])
        results.append([])
        for i_idx, row in enumerate(rows_s):
            celeb = str(row.get("celebrity_name", "")).strip()
            pro = str(row.get("ballroom_partner", "")).strip()
            name = f"{celeb} / {pro}" if celeb or pro else f"Couple_{i_idx+1}"
            couple_names[s_idx].append(name)
            results[s_idx].append(str(row.get("results", "")).strip())
            try:
                placement[s_idx, i_idx] = float(row.get("placement", ""))
            except Exception:
                placement[s_idx, i_idx] = np.nan

            for week in range(1, T + 1):
                scores: List[float] = []
                any_score = False
                for judge in range(1, 5):
                    col = mapping.get((week, judge))
                    raw = row.get(col) if col else None
                    score = _parse_score(raw)
                    if score is not None:
                        any_score = True
                        scores.append(score)
                if any_score:
                    has_any[s_idx, week - 1, i_idx] = True
                    J_obs[s_idx, week - 1, i_idx] = float(sum(scores))

        # pad names/results
        while len(couple_names[s_idx]) < N_max:
            couple_names[s_idx].append("")
            results[s_idx].append("")

    return Dataset(
        seasons=seasons,
        T_max=T,
        N_max=N_max,
        couple_names=couple_names,
        J_obs=J_obs,
        has_any_score=has_any,
        results=results,
        placement=placement,
    )


def _parse_elim_week(results: str) -> int | None:
    if not results:
        return None
    m = re.match(r"Eliminated Week (\d+)", results.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def build_masks(dataset: Dataset, withdrawal_policy: str = "exclude_only_withdrawn", multi_elim_policy: str = "topk"):
    S, T, N = dataset.J_obs.shape
    active_mask = np.zeros((S, T, N), dtype=bool)
    elim_mask = np.zeros((S, T, N), dtype=bool)
    withdraw_mask = np.zeros((S, T, N), dtype=bool)
    valid_mask = np.zeros((S, T), dtype=bool)

    for s in range(S):
        # first zero week for each contestant
        first_zero = np.full((N,), fill_value=-1, dtype=int)
        for i in range(N):
            has_any = dataset.has_any_score[s, :, i]
            zeros = np.where(has_any & (dataset.J_obs[s, :, i] == 0.0))[0]
            if zeros.size > 0:
                first_zero[i] = int(zeros.min()) + 1  # week number

        for i in range(N):
            for t in range(T):
                if not dataset.has_any_score[s, t, i]:
                    continue
                if dataset.J_obs[s, t, i] <= 0:
                    continue
                fz = first_zero[i]
                if fz > 0 and (t + 1) >= fz:
                    continue
                active_mask[s, t, i] = True

        # elimination from results
        for i in range(N):
            res = dataset.results[s][i] if i < len(dataset.results[s]) else ""
            if not res:
                continue
            if res.strip() == "Withdrew":
                continue
            elim_week = _parse_elim_week(res)
            if elim_week is None:
                continue
            if 1 <= elim_week <= T:
                elim_mask[s, elim_week - 1, i] = True

        # withdrawal handling
        for i in range(N):
            res = dataset.results[s][i] if i < len(dataset.results[s]) else ""
            if res.strip() != "Withdrew":
                continue
            # infer withdrawal week as first week with no score after being active
            has_any = dataset.has_any_score[s, :, i]
            active_weeks = np.where(has_any)[0]
            if active_weeks.size == 0:
                continue
            last_active = int(active_weeks.max())
            wd_week = min(last_active + 2, T)  # week number after last active
            if 1 <= wd_week <= T:
                withdraw_mask[s, wd_week - 1, i] = True

        # valid weeks
        for t in range(T):
            n_elim = int(elim_mask[s, t].sum())
            has_any = bool(dataset.has_any_score[s, t].any())
            valid = (n_elim >= 1) and has_any
            if multi_elim_policy == "invalidate_week" and n_elim > 1:
                valid = False
            if withdrawal_policy == "invalidate_week" and withdraw_mask[s, t].any():
                valid = False
            valid_mask[s, t] = valid

    multi_elim_count = elim_mask.sum(axis=2).astype(int)
    active_init = active_mask[:, 0, :].copy()
    return {
        "active_mask": active_mask,
        "valid_mask": valid_mask,
        "elim_mask": elim_mask,
        "withdraw_mask": withdraw_mask,
        "multi_elim_count": multi_elim_count,
        "active_init": active_init,
    }
