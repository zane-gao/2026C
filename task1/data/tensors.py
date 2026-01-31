from __future__ import annotations

from typing import Dict, List, Tuple

from ..types import RuleType, TensorPack, WeekObs


def _rule_for_season(season: int, season_rules):
    for start, end, rule in season_rules:
        if start <= season <= end:
            return rule
    raise ValueError(f"No rule for season {season}")


def build_tensors(long_df, valid_df, features_df, feature_cols, season_rules, max_week: int):
    try:
        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas/numpy required for build_tensors") from exc

    seasons = sorted(long_df["season"].unique().tolist())
    S = len(seasons)

    # couple meta table
    couple_meta = long_df[
        ["couple_id", "season", "celebrity_name", "ballroom_partner"]
    ].drop_duplicates(subset=["couple_id"]).copy()

    # global ids
    celeb_names = sorted(couple_meta["celebrity_name"].dropna().unique().tolist())
    pro_names = sorted(couple_meta["ballroom_partner"].dropna().unique().tolist())
    celeb_id_map = {n: i for i, n in enumerate(celeb_names)}
    pro_id_map = {n: i for i, n in enumerate(pro_names)}

    couple_meta["celebrity_id"] = couple_meta["celebrity_name"].map(celeb_id_map).fillna(-1).astype(int)
    couple_meta["pro_id"] = couple_meta["ballroom_partner"].map(pro_id_map).fillna(-1).astype(int)

    # per season max N
    season_couples = {
        s: couple_meta.loc[couple_meta["season"] == s, "couple_id"].tolist() for s in seasons
    }
    N_max = max(len(v) for v in season_couples.values())
    T_max = max_week

    # allocate tensors
    X = np.zeros((S, N_max, len(feature_cols)), dtype=float)
    J = np.zeros((S, T_max, N_max), dtype=float)
    J_z = np.zeros((S, T_max, N_max), dtype=float)
    J_pct = np.zeros((S, T_max, N_max), dtype=float)
    J_cum_avg = np.zeros((S, T_max, N_max), dtype=float)
    active_mask = np.zeros((S, T_max, N_max), dtype=float)
    elim_mask = np.zeros((S, T_max, N_max), dtype=float)
    valid_mask = np.zeros((S, T_max), dtype=bool)
    rule_id = np.zeros((S, T_max), dtype=int)
    pro_id = -np.ones((S, N_max), dtype=int)
    celeb_id = -np.ones((S, N_max), dtype=int)
    couple_names: List[List[str]] = []

    # build map from couple_id to features
    feat_map = features_df.set_index("couple_id")

    for s_idx, s in enumerate(seasons):
        couples = season_couples[s]
        couples_sorted = sorted(couples)
        couple_names.append([])

        # week validity and rule
        rule = _rule_for_season(s, season_rules)
        rule_val = RuleType(rule)
        for t in range(1, T_max + 1):
            rule_id[s_idx, t - 1] = list(RuleType).index(rule_val)

        valid_rows = valid_df.loc[valid_df["season"] == s]
        valid_map = {(int(r.week)): bool(r.valid) for r in valid_rows.itertuples()}
        for t in range(1, T_max + 1):
            valid_mask[s_idx, t - 1] = bool(valid_map.get(t, False))

        # prepare couple-level mapping
        for i_idx, cid in enumerate(couples_sorted):
            meta = couple_meta.loc[couple_meta["couple_id"] == cid].iloc[0]
            name = f"{meta['celebrity_name']} / {meta['ballroom_partner']}"
            couple_names[s_idx].append(name)
            pro_id[s_idx, i_idx] = int(meta["pro_id"])
            celeb_id[s_idx, i_idx] = int(meta["celebrity_id"])

            if cid in feat_map.index:
                X[s_idx, i_idx, :] = feat_map.loc[cid, feature_cols].values.astype(float)

            # fill week data
            sub = long_df[(long_df["couple_id"] == cid)].copy()
            for t in range(1, T_max + 1):
                row = sub[sub["week"] == t]
                if row.empty:
                    continue
                r = row.iloc[0]
                val = float(r["J"]) if r["J"] is not None else 0.0
                if not np.isfinite(val):
                    val = 0.0
                J[s_idx, t - 1, i_idx] = val
                active_mask[s_idx, t - 1, i_idx] = 1.0 if r["is_active"] else 0.0
                elim_mask[s_idx, t - 1, i_idx] = 1.0 if r["is_elim"] else 0.0

        # compute J_z and J_pct per week within season
        for t in range(1, T_max + 1):
            a = active_mask[s_idx, t - 1, :].astype(bool)
            if not a.any():
                continue
            j_vals = J[s_idx, t - 1, a]
            mean = float(j_vals.mean())
            std = float(j_vals.std())
            if std <= 0:
                std = 1.0
            J_z[s_idx, t - 1, a] = (j_vals - mean) / std
            total = float(j_vals.sum())
            if total > 0:
                J_pct[s_idx, t - 1, a] = j_vals / total

        # cumulative avg J
        for i_idx, cid in enumerate(couples_sorted):
            cum_sum = 0.0
            cum_cnt = 0
            for t in range(1, T_max + 1):
                if active_mask[s_idx, t - 1, i_idx] > 0:
                    cum_sum += J[s_idx, t - 1, i_idx]
                    cum_cnt += 1
                J_cum_avg[s_idx, t - 1, i_idx] = (cum_sum / cum_cnt) if cum_cnt > 0 else 0.0

        # pad couple_names to N_max for alignment
        if len(couple_names[s_idx]) < N_max:
            couple_names[s_idx].extend([""] * (N_max - len(couple_names[s_idx])))

    pack = TensorPack(
        seasons=seasons,
        T_max=T_max,
        N_max=N_max,
        X=X,
        J=J,
        J_z=J_z,
        J_pct=J_pct,
        J_cum_avg=J_cum_avg,
        active_mask=active_mask,
        elim_mask=elim_mask,
        valid_mask=valid_mask,
        rule_id=rule_id,
        pro_id=pro_id,
        celeb_id=celeb_id,
        couple_names=couple_names,
        feature_names=feature_cols,
    )
    return pack


def build_week_obs_list(tensors: TensorPack):
    obs_list: List[WeekObs] = []
    for s_idx, season in enumerate(tensors.seasons):
        for t in range(tensors.T_max):
            rule = list(RuleType)[tensors.rule_id[s_idx, t]]
            obs = WeekObs(
                season=season,
                week=t + 1,
                rule=rule,
                valid=bool(tensors.valid_mask[s_idx, t]),
                active_mask=tensors.active_mask[s_idx, t, :],
                elim_mask=tensors.elim_mask[s_idx, t, :],
                J=tensors.J[s_idx, t, :],
                J_z=tensors.J_z[s_idx, t, :],
                J_pct=tensors.J_pct[s_idx, t, :],
                J_cum_avg=tensors.J_cum_avg[s_idx, t, :],
            )
            obs_list.append(obs)
    return obs_list
