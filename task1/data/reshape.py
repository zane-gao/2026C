from __future__ import annotations

import math
import re
from typing import Dict, Tuple


def _parse_score(val):
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


def detect_week_columns(df) -> Tuple[Dict[Tuple[int, int], str], int]:
    pattern = re.compile(r"^week(\d+)_judge(\d+)_score$")
    mapping: Dict[Tuple[int, int], str] = {}
    max_week = 0
    for col in df.columns:
        m = pattern.match(col)
        if not m:
            continue
        week = int(m.group(1))
        judge = int(m.group(2))
        mapping[(week, judge)] = col
        if week > max_week:
            max_week = week
    return mapping, max_week


def wide_to_long(df, max_week: int | None = None):
    try:
        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas/numpy required for wide_to_long") from exc

    mapping, detected_max = detect_week_columns(df)
    if max_week is None:
        max_week = detected_max

    df = df.copy()
    df["couple_id"] = list(range(len(df)))

    static_cols = [
        "couple_id",
        "celebrity_name",
        "ballroom_partner",
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "celebrity_age_during_season",
        "season",
        "results",
        "placement",
    ]

    records = []
    for _, row in df.iterrows():
        for week in range(1, max_week + 1):
            scores = []
            n_judges = 0
            has_any = False
            score_cols = {}
            for judge in range(1, 5):
                col = mapping.get((week, judge))
                raw = row.get(col) if col else None
                score = _parse_score(raw)
                score_cols[f"judge{judge}_score"] = score
                if score is not None:
                    has_any = True
                    n_judges += 1
                    scores.append(score)
            total = float(sum(scores)) if has_any else 0.0

            rec = {col: row.get(col) for col in static_cols}
            rec.update(score_cols)
            rec.update(
                {
                    "week": week,
                    "has_any_score": bool(has_any),
                    "n_judges": int(n_judges),
                    "J": total,
                }
            )
            records.append(rec)

    long_df = pd.DataFrame.from_records(records)

    # coerce numeric columns
    long_df["season"] = long_df["season"].astype(int)
    long_df["week"] = long_df["week"].astype(int)
    long_df["celebrity_age_during_season"] = pd.to_numeric(
        long_df["celebrity_age_during_season"], errors="coerce"
    )
    long_df["placement"] = pd.to_numeric(long_df["placement"], errors="coerce")

    return long_df
