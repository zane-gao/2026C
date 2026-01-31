from __future__ import annotations

import re
from typing import Dict, Tuple


def _parse_elim_week(results: str | None) -> int | None:
    if results is None:
        return None
    m = re.match(r"Eliminated Week (\d+)", str(results).strip())
    if not m:
        return None
    return int(m.group(1))


def build_masks(long_df, withdrawal_policy: str = "exclude_only_withdrawn", multi_elim_policy: str = "topk"):
    try:
        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas/numpy required for build_masks") from exc

    df = long_df.copy()
    df["elim_week"] = df["results"].apply(_parse_elim_week)
    df["is_withdrew"] = df["results"].fillna("") == "Withdrew"

    # first zero week among weeks with any score
    zero_week = (
        df.loc[df["has_any_score"] & (df["J"] == 0.0)]
        .groupby("couple_id")["week"]
        .min()
    )
    df["first_zero_week"] = df["couple_id"].map(zero_week)

    # active: positive score and before first zero week (if exists)
    def _is_active(row):
        if not row["has_any_score"]:
            return False
        if row["J"] <= 0:
            return False
        fzw = row["first_zero_week"]
        if pd.isna(fzw):
            return True
        return row["week"] < int(fzw)

    df["is_active"] = df.apply(_is_active, axis=1)

    # elimination flag from results
    df["is_elim"] = df["elim_week"].notna() & (df["week"] == df["elim_week"])

    # withdrawal handling: do not mark as elimination
    df.loc[df["is_withdrew"], "is_elim"] = False

    # valid week: at least one elimination by default
    elim_counts = (
        df.groupby(["season", "week"])["is_elim"].sum().rename("n_elim")
    )
    week_has_any = (
        df.groupby(["season", "week"])["has_any_score"].any().rename("has_any")
    )

    valid = elim_counts.to_frame().join(week_has_any).reset_index()
    valid["valid"] = (valid["n_elim"] >= 1) & (valid["has_any"])

    if multi_elim_policy == "invalidate_week":
        valid.loc[valid["n_elim"] > 1, "valid"] = False

    if withdrawal_policy == "invalidate_week":
        withdraw_counts = (
            df.groupby(["season", "week"])["is_withdrew"].sum().rename("n_withdrew")
        )
        valid = valid.merge(withdraw_counts.reset_index(), on=["season", "week"], how="left")
        valid["n_withdrew"] = valid["n_withdrew"].fillna(0)
        # any withdrawal invalidates this week
        valid.loc[valid["n_withdrew"] > 0, "valid"] = False

    return df, valid
