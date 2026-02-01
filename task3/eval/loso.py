from __future__ import annotations

from typing import Dict
import numpy as np

from .metrics import r2_rmse, auc_score, brier_score


def loso_linear(df, response: str, formula: str, season_col: str = "season") -> Dict:
    try:
        import statsmodels.formula.api as smf  # type: ignore
    except Exception as exc:
        raise RuntimeError("statsmodels is required for LOSO") from exc

    metrics = []
    seasons = sorted(df[season_col].unique())
    for season in seasons:
        train = df[df[season_col] != season]
        test = df[df[season_col] == season]
        if train.empty or test.empty:
            continue
        model = smf.ols(formula, data=train).fit()
        pred = model.predict(test)
        metrics.append(r2_rmse(test[response].to_numpy(), pred))

    if not metrics:
        return {"r2": float("nan"), "rmse": float("nan")}
    r2_vals = [m["r2"] for m in metrics]
    rmse_vals = [m["rmse"] for m in metrics]
    return {
        "r2": float(np.nanmean(r2_vals)),
        "rmse": float(np.nanmean(rmse_vals)),
    }


def loso_logistic(df, response: str, formula: str, season_col: str = "season") -> Dict:
    try:
        import statsmodels.api as sm  # type: ignore
        import statsmodels.formula.api as smf  # type: ignore
    except Exception as exc:
        raise RuntimeError("statsmodels is required for LOSO") from exc

    metrics = []
    seasons = sorted(df[season_col].unique())
    for season in seasons:
        train = df[df[season_col] != season]
        test = df[df[season_col] == season]
        if train.empty or test.empty:
            continue
        model = smf.glm(formula, data=train, family=sm.families.Binomial()).fit()
        pred = model.predict(test)
        auc = auc_score(test[response].to_numpy(), pred)
        brier = brier_score(test[response].to_numpy(), pred)
        metrics.append({"auc": auc, "brier": brier})

    if not metrics:
        return {"auc": float("nan"), "brier": float("nan")}
    return {
        "auc": float(np.nanmean([m["auc"] for m in metrics])),
        "brier": float(np.nanmean([m["brier"] for m in metrics])),
    }
