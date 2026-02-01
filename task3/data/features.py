from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..types import FeatureSpec


def _winsorize(series, q: float):
    if series is None:
        return series
    lower = series.quantile(1.0 - q)
    upper = series.quantile(q)
    return series.clip(lower, upper)


def build_feature_spec(panel, include_platform: bool = False) -> FeatureSpec:
    base_numeric = ["age"]
    base_categorical = ["industry", "homestate", "homecountry"]
    social_cols = ["P_cele", "P_partner"]
    missing_cols = ["missing_cele_total", "missing_partner_total"]

    platform_cols: List[str] = []
    if include_platform:
        for col in panel.columns:
            if col.startswith("P_cele_") or col.startswith("P_partner_"):
                platform_cols.append(col)
            if col.startswith("missing_cele_") or col.startswith("missing_partner_"):
                missing_cols.append(col)
    return FeatureSpec(
        base_numeric=base_numeric,
        base_categorical=base_categorical,
        social_cols=social_cols,
        missing_cols=missing_cols,
        platform_cols=platform_cols,
    )


def prepare_features(panel, include_social: bool = True, include_platform: bool = False,
                     missing_as_zero: bool = False, winsorize_q: float | None = None) -> Tuple["object", FeatureSpec]:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required to prepare features") from exc

    df = panel.copy()
    df["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    if df["age"].isna().all():
        df["age"] = 0.0
    else:
        df["age"] = df["age"].fillna(df["age"].median())

    for col in ["industry", "homestate", "homecountry"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
        df[col] = df[col].replace({"": "Unknown"})

    for col in ["P_cele", "P_partner"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if winsorize_q is not None:
        for col in ["P_cele", "P_partner"]:
            df[col] = _winsorize(df[col], winsorize_q)

    if "missing_cele_total" not in df.columns:
        df["missing_cele_total"] = df["P_cele"].isna()
    if "missing_partner_total" not in df.columns:
        df["missing_partner_total"] = df["P_partner"].isna()

    # Always impute social features to avoid dropping rows in formula handling.
    df["P_cele"] = df["P_cele"].fillna(0.0)
    df["P_partner"] = df["P_partner"].fillna(0.0)
    if missing_as_zero:
        df["missing_cele_total"] = False
        df["missing_partner_total"] = False

    if include_platform:
        for col in df.columns:
            if col.startswith("P_cele_") or col.startswith("P_partner_"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if winsorize_q is not None:
                    df[col] = _winsorize(df[col], winsorize_q)
                df[col] = df[col].fillna(0.0)

    spec = build_feature_spec(df, include_platform=include_platform)
    if not include_social:
        spec.social_cols = []
        spec.platform_cols = []
        spec.missing_cols = []

    return df, spec
