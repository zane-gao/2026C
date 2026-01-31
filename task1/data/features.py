from __future__ import annotations

from typing import List, Tuple


def build_features(long_df):
    try:
        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas/numpy required for build_features") from exc

    # one row per couple (season-level)
    cols = [
        "couple_id",
        "season",
        "celebrity_industry",
        "celebrity_homecountry/region",
        "celebrity_age_during_season",
        "celebrity_homestate",
        "ballroom_partner",
        "celebrity_name",
    ]
    df = long_df[cols].drop_duplicates(subset=["couple_id"]).copy()

    df["age"] = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce")
    age_mean = df["age"].mean()
    if not np.isfinite(age_mean):
        age_mean = 0.0
    age_std = df["age"].std()
    if not np.isfinite(age_std) or age_std <= 0:
        age_std = 1.0
    df["age_z"] = (df["age"].fillna(age_mean) - age_mean) / age_std

    df["homecountry_is_us"] = (
        df["celebrity_homecountry/region"].fillna("") == "United States"
    ).astype(float)

    # industry one-hot
    industry = df["celebrity_industry"].fillna("Unknown")
    industry_dummies = pd.get_dummies(industry, prefix="industry")

    feats = pd.concat([df[["couple_id", "season", "age_z", "homecountry_is_us"]], industry_dummies], axis=1)

    feature_cols = [c for c in feats.columns if c not in ("couple_id", "season")]
    return feats, feature_cols
