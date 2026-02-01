from __future__ import annotations

from typing import Dict, Optional
import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    uniq, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    for u_idx, count in enumerate(counts):
        if count <= 1:
            continue
        idx = np.where(inv == u_idx)[0]
        ranks[idx] = ranks[idx].mean()
    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    rx = _rankdata(x[mask])
    ry = _rankdata(y[mask])
    return float(np.corrcoef(rx, ry)[0, 1])


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def build_assortative_mating_report(panel, u_pro_df=None, v_celeb_df=None) -> Dict:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required for assortative mating") from exc

    df = panel.copy()
    df = df[df["is_active"]]
    df = df.sort_values(["season", "couple_id", "week"])
    first = df.groupby(["season", "couple_id"], as_index=False).first()

    report: Dict[str, object] = {}
    # couple-level correlation
    report["couple_Jz_Pcele_corr"] = pearsonr(first["J_z"].to_numpy(), first["P_cele"].to_numpy())
    report["couple_Jz_Pcele_spearman"] = spearmanr(first["J_z"].to_numpy(), first["P_cele"].to_numpy())

    pro_base = first.groupby("pro_id").agg({
        "J_z": "mean",
        "P_cele": "mean",
        "celeb_id": "nunique",
    }).rename(columns={"celeb_id": "n_celeb"})
    report["pro_Jz_Pcele_corr"] = pearsonr(pro_base["J_z"].to_numpy(), pro_base["P_cele"].to_numpy())

    if u_pro_df is not None and not u_pro_df.empty:
        merged = pro_base.merge(u_pro_df[["pro_id", "u_pro"]], on="pro_id", how="left")
        report["u_pro_Pcele_corr"] = pearsonr(merged["u_pro"].to_numpy(), merged["P_cele"].to_numpy())
        report["u_pro_Jz_corr"] = pearsonr(merged["u_pro"].to_numpy(), merged["J_z"].to_numpy())

    if v_celeb_df is not None and not v_celeb_df.empty:
        celeb_base = first.groupby("celeb_id").agg({
            "J_z": "mean",
            "P_cele": "mean",
        })
        if "v_celeb" in v_celeb_df.columns:
            use_col = "v_celeb"
        elif "median" in v_celeb_df.columns:
            use_col = "median"
        else:
            use_col = None
        if use_col is not None:
            merged_c = celeb_base.merge(v_celeb_df[["celeb_id", use_col]], on="celeb_id", how="left")
            report["v_celeb_Pcele_corr"] = pearsonr(merged_c[use_col].to_numpy(), merged_c["P_cele"].to_numpy())

    multi_pro = pro_base[pro_base["n_celeb"] >= 2]
    report["multi_pro_count"] = int(multi_pro.shape[0])
    if multi_pro.shape[0] >= 2:
        report["multi_pro_Jz_Pcele_corr"] = pearsonr(multi_pro["J_z"].to_numpy(), multi_pro["P_cele"].to_numpy())

    return report
