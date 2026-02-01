from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task3.io.export import load_parquet
from task3.viz import fig_pro_forest, fig_pro_quadrant, fig_var_decomp, fig_delta_beta, fig_platform_specificity, fig_shap


def _load_if_exists(path: Path):
    """加载文件如果存在."""
    if path.exists():
        return load_parquet(path)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Task3: postprocess outputs")
    parser.add_argument("--run", type=str, required=True, help="Run directory")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        raise FileNotFoundError(str(run_dir))

    # 尝试加载 PyTorch 或 statsmodels 输出
    # PyTorch 输出
    m1_torch_fixed = _load_if_exists(run_dir / "m1_torch_fixed.parquet")
    m1_torch_pro = _load_if_exists(run_dir / "m1_torch_random_pro.parquet")
    m2_torch_fixed = _load_if_exists(run_dir / "m2_torch_fixed.parquet")
    m2_torch_pro = _load_if_exists(run_dir / "m2_torch_random_pro.parquet")
    m1_torch_var = None
    if (run_dir / "m1_torch_var_components.json").exists():
        m1_torch_var = json.loads((run_dir / "m1_torch_var_components.json").read_text(encoding="utf-8"))
    
    # statsmodels 输出
    m1_fixed = _load_if_exists(run_dir / "m1_judges_fixed.parquet")
    m2_fixed = _load_if_exists(run_dir / "m2_fans_fixed.parquet")
    m1_random = _load_if_exists(run_dir / "m1_judges_random.parquet")
    m2_random = _load_if_exists(run_dir / "m2_fans_random.parquet")

    # 优先使用 PyTorch 输出
    use_torch = m1_torch_fixed is not None

    # Pro forest plots
    if use_torch and m1_torch_pro is not None:
        pro_j = m1_torch_pro.copy()
        pro_j["median"] = pro_j["u_pro"]
        pro_j["q05"] = pro_j["u_pro"]
        pro_j["q95"] = pro_j["u_pro"]
        fig_pro_forest.plot(run_dir / "fig_pro_forest_j.png", {"pro_effects": pro_j}, channel="J")
    elif m1_random is not None:
        pro_j = m1_random[m1_random["entity_type"] == "pro"].copy()
        pro_j = pro_j.rename(columns={"effect": "median"})
        pro_j["q05"] = pro_j["median"]
        pro_j["q95"] = pro_j["median"]
        fig_pro_forest.plot(run_dir / "fig_pro_forest_j.png", {"pro_effects": pro_j}, channel="J")

    if use_torch and m2_torch_pro is not None:
        pro_f = m2_torch_pro.copy()
        fig_pro_forest.plot(run_dir / "fig_pro_forest_f.png", {"pro_effects": pro_f}, channel="F")
    elif m2_random is not None:
        pro_f = m2_random[m2_random["entity_type"] == "pro"].copy()
        fig_pro_forest.plot(run_dir / "fig_pro_forest_f.png", {"pro_effects": pro_f}, channel="F")

    # Pro quadrant
    if use_torch and m1_torch_pro is not None and m2_torch_pro is not None:
        pro_j = m1_torch_pro.copy()
        pro_j = pro_j.rename(columns={"u_pro": "u_pro_J"})
        pro_f = m2_torch_pro.copy()
        pro_f = pro_f.rename(columns={"median": "u_pro_F"})
        merged = pro_j.merge(pro_f, on="pro_id", how="inner", suffixes=("_j", "_f"))
        fig_pro_quadrant.plot(run_dir / "fig_pro_quadrant.png", {"pro_quadrant": merged[["u_pro_J", "u_pro_F"]]})
    elif m1_random is not None and m2_random is not None:
        pro_j = m1_random[m1_random["entity_type"] == "pro"].copy()
        pro_j = pro_j.rename(columns={"effect": "u_pro_J"})
        pro_f = m2_random[m2_random["entity_type"] == "pro"].copy()
        pro_f = pro_f.rename(columns={"median": "u_pro_F"})
        merged = pro_j.merge(pro_f, left_on="entity_id", right_on="entity_id", how="inner", suffixes=("_j", "_f"))
        fig_pro_quadrant.plot(run_dir / "fig_pro_quadrant.png", {"pro_quadrant": merged[["u_pro_J", "u_pro_F"]]})

    # Variance decomposition
    var_j = {}
    var_f = {}
    
    # 加载 M2 方差分量
    m2_torch_var = None
    if (run_dir / "m2_torch_var_components.json").exists():
        m2_torch_var = json.loads((run_dir / "m2_torch_var_components.json").read_text(encoding="utf-8"))
    
    if use_torch and m1_torch_var is not None:
        var_j = m1_torch_var
    elif m1_random is not None:
        var_rows = m1_random[m1_random["entity_type"] == "var_component"]
        for _, row in var_rows.iterrows():
            var_j[str(row["entity_id"])] = float(row["effect"])
    
    if use_torch and m2_torch_var is not None:
        var_f = m2_torch_var
    
    fig_var_decomp.plot(run_dir / "fig_var_decomp.png", {"var_j": var_j, "var_f": var_f})

    # Delta beta social
    fixed_to_use = m1_torch_fixed if use_torch else m1_fixed
    fixed_f_to_use = m2_torch_fixed if use_torch else m2_fixed
    if fixed_to_use is not None and fixed_f_to_use is not None:
        social_terms = [t for t in fixed_f_to_use["term"] if "P_" in t or "missing" in t]
        rows = []
        for term in social_terms:
            val_f = fixed_f_to_use[fixed_f_to_use["term"] == term]["median"].iloc[0] if "median" in fixed_f_to_use.columns else 0.0
            val_j = 0.0
            if fixed_to_use is not None and term in set(fixed_to_use["term"]):
                val_j = fixed_to_use[fixed_to_use["term"] == term]["estimate"].iloc[0]
            rows.append({"term": term, "delta": float(val_f - val_j)})
        if rows:
            import pandas as pd  # type: ignore
            df = pd.DataFrame(rows)
            fig_delta_beta.plot(run_dir / "fig_delta_beta_social.png", {"delta_beta": df})

    # Platform specificity (optional)
    if fixed_f_to_use is not None:
        rows = []
        for _, row in fixed_f_to_use.iterrows():
            term = str(row["term"])
            if "P_cele_" in term or "P_partner_" in term:
                val = row.get("median", row.get("estimate", 0.0))
                rows.append({"platform": term, "estimate": float(val)})
        if rows:
            import pandas as pd  # type: ignore
            df = pd.DataFrame(rows)
            fig_platform_specificity.plot(run_dir / "fig_platform_specificity.png", {"platform_effects": df})

    # SHAP / Feature importance plot
    fixed_for_shap = m1_torch_fixed if use_torch else m1_fixed
    if fixed_for_shap is not None:
        fig_shap.plot(run_dir / "fig_shap_residual.png", {"fixed_effects": fixed_for_shap})
    else:
        fig_shap.plot(run_dir / "fig_shap_residual.png", {})

    # ========================================
    # 生成 task3_artifact.json（给 Task2/4/5 的接口产物）
    # ========================================
    artifact = {
        "description": "Task3 artifact for downstream tasks (Task2/4/5)",
        "pro_effects": {},
        "celeb_effects": {},
        "fixed_effects": {},
        "metrics": {},
        "social_coefficients": {},
    }
    
    # Pro 效应 (u_pro_J, u_pro_F)
    if use_torch and m1_torch_pro is not None:
        for _, row in m1_torch_pro.iterrows():
            pid = str(row["pro_id"])
            artifact["pro_effects"][pid] = artifact["pro_effects"].get(pid, {})
            artifact["pro_effects"][pid]["u_pro_J"] = float(row["u_pro"])
    
    if use_torch and m2_torch_pro is not None:
        for _, row in m2_torch_pro.iterrows():
            pid = str(row["pro_id"])
            artifact["pro_effects"][pid] = artifact["pro_effects"].get(pid, {})
            artifact["pro_effects"][pid]["u_pro_F"] = float(row["median"])
            artifact["pro_effects"][pid]["u_pro_F_q05"] = float(row.get("q05", row["median"]))
            artifact["pro_effects"][pid]["u_pro_F_q95"] = float(row.get("q95", row["median"]))
    
    # Celeb 效应
    m1_torch_celeb = _load_if_exists(run_dir / "m1_torch_random_celeb.parquet")
    m2_torch_celeb = _load_if_exists(run_dir / "m2_torch_random_celeb.parquet")
    
    if use_torch and m1_torch_celeb is not None:
        for _, row in m1_torch_celeb.iterrows():
            cid = str(row["celeb_id"])
            artifact["celeb_effects"][cid] = artifact["celeb_effects"].get(cid, {})
            artifact["celeb_effects"][cid]["v_celeb_J"] = float(row["v_celeb"])
    
    if use_torch and m2_torch_celeb is not None:
        for _, row in m2_torch_celeb.iterrows():
            cid = str(row["celeb_id"])
            artifact["celeb_effects"][cid] = artifact["celeb_effects"].get(cid, {})
            artifact["celeb_effects"][cid]["v_celeb_F"] = float(row["median"])
    
    # 固定效应（关键系数）
    if fixed_to_use is not None:
        key_terms = ["age", "P_cele", "P_partner", "missing_cele_total", "missing_partner_total", "week"]
        for term in key_terms:
            match = fixed_to_use[fixed_to_use["term"] == term]
            if len(match) > 0:
                artifact["fixed_effects"][f"beta_J_{term}"] = float(match.iloc[0]["estimate"])
    
    if fixed_f_to_use is not None:
        key_terms = ["age", "P_cele", "P_partner", "missing_cele_total", "missing_partner_total", "week"]
        for term in key_terms:
            match = fixed_f_to_use[fixed_f_to_use["term"] == term]
            if len(match) > 0:
                col = "median" if "median" in fixed_f_to_use.columns else "estimate"
                artifact["fixed_effects"][f"beta_F_{term}"] = float(match.iloc[0][col])
    
    # 社媒系数差异 (Delta beta)
    if fixed_to_use is not None and fixed_f_to_use is not None:
        social_terms = ["P_cele", "P_partner"]
        for term in social_terms:
            match_j = fixed_to_use[fixed_to_use["term"] == term]
            match_f = fixed_f_to_use[fixed_f_to_use["term"] == term]
            if len(match_j) > 0 and len(match_f) > 0:
                val_j = float(match_j.iloc[0]["estimate"])
                col_f = "median" if "median" in fixed_f_to_use.columns else "estimate"
                val_f = float(match_f.iloc[0][col_f])
                artifact["social_coefficients"][term] = {
                    "beta_J": val_j,
                    "beta_F": val_f,
                    "delta": val_f - val_j,
                }
    
    # 加载 metrics
    m1_metrics_path = run_dir / "m1_torch_metrics.json"
    m2_metrics_path = run_dir / "m2_torch_metrics.json"
    m3_metrics_path = run_dir / "m3_torch_metrics.json"
    
    if m1_metrics_path.exists():
        artifact["metrics"]["M1"] = json.loads(m1_metrics_path.read_text(encoding="utf-8"))
    if m2_metrics_path.exists():
        artifact["metrics"]["M2"] = json.loads(m2_metrics_path.read_text(encoding="utf-8"))
    if m3_metrics_path.exists():
        m3_data = json.loads(m3_metrics_path.read_text(encoding="utf-8"))
        artifact["metrics"]["M3"] = m3_data
        # 保存关键的 eta 系数
        artifact["fixed_effects"]["eta_J"] = m3_data.get("eta_J", 0.0)
        artifact["fixed_effects"]["eta_F"] = m3_data.get("eta_F", 0.0)
    
    # 方差分解
    artifact["variance_decomposition"] = {
        "judges": var_j,
        "fans": var_f,
    }
    
    # 保存 artifact
    artifact_path = run_dir / "task3_artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved task3_artifact.json")

    print(f"Saved figures to {run_dir}")


if __name__ == "__main__":
    main()
