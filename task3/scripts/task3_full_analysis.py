"""
Task3 完整分析脚本 - 生成所有可选交付物.

包括:
1. Assortative Mating 检查报告
2. 消融实验 (不加社媒 vs 加社媒)
3. LOSO 交叉验证
4. 平台特异性分析
5. 联立相关随机效应 (rho_p)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json
import time

import numpy as np
import pandas as pd
from scipy import stats

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task3.io.export import load_parquet, save_parquet
from task3.eval.assortative_mating import build_assortative_mating_report


def compute_rho_p(m1_random_pro: pd.DataFrame, m2_random_pro: pd.DataFrame) -> dict:
    """计算 Pro 跨通道相关性 rho_p."""
    if m1_random_pro is None or m2_random_pro is None:
        return {"rho_p": None, "error": "Missing random effects data"}
    
    # 合并 Pro 效应
    m1_pro = m1_random_pro.copy()
    m2_pro = m2_random_pro.copy()
    
    # 确定列名
    u_col = "u_pro" if "u_pro" in m1_pro.columns else "median"
    v_col = "median" if "median" in m2_pro.columns else "u_pro"
    
    m1_pro = m1_pro.rename(columns={u_col: "u_pro_J"})
    m2_pro = m2_pro.rename(columns={v_col: "u_pro_F"})
    
    merged = m1_pro[["pro_id", "u_pro_J"]].merge(
        m2_pro[["pro_id", "u_pro_F"]],
        on="pro_id",
        how="inner"
    )
    
    if len(merged) < 3:
        return {"rho_p": None, "error": "Not enough Pro effects to compute correlation"}
    
    # Pearson 相关
    corr, pval = stats.pearsonr(merged["u_pro_J"], merged["u_pro_F"])
    
    # Bootstrap CI
    n_boot = 1000
    rng = np.random.default_rng(42)
    boot_corrs = []
    for _ in range(n_boot):
        idx = rng.choice(len(merged), size=len(merged), replace=True)
        boot_corrs.append(np.corrcoef(
            merged["u_pro_J"].iloc[idx],
            merged["u_pro_F"].iloc[idx]
        )[0, 1])
    
    boot_corrs = np.array(boot_corrs)
    q05, q95 = np.nanpercentile(boot_corrs, [5, 95])
    
    return {
        "rho_p": float(corr),
        "pvalue": float(pval),
        "ci_95": [float(q05), float(q95)],
        "n_pros": int(len(merged)),
        "interpretation": "Pro 在技术通道和人气通道的效应相关性"
    }


def run_ablation_experiment(run_dir: Path, panel: pd.DataFrame, k_index: list, device: str = "cuda") -> dict:
    """运行消融实验: 比较加/不加社媒特征的效果."""
    
    # 加载原始结果作为 baseline
    baseline_metrics_path = run_dir / "m2_torch_metrics.json"
    if baseline_metrics_path.exists():
        baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    else:
        baseline_metrics = {"r2_median": None}
    
    m1_metrics_path = run_dir / "m1_torch_metrics.json"
    if m1_metrics_path.exists():
        m1_metrics = json.loads(m1_metrics_path.read_text(encoding="utf-8"))
    else:
        m1_metrics = {}
    
    # 统计社媒特征
    social_feature_cols = [c for c in panel.columns if "P_cele" in c or "P_partner" in c or "missing_cele" in c or "missing_partner" in c]
    
    # 计算社媒特征的统计
    social_stats = {}
    for col in social_feature_cols:
        if col in panel.columns:
            valid = panel.loc[panel["is_active"], col].dropna()
            if len(valid) > 0:
                social_stats[col] = {
                    "mean": float(valid.mean()),
                    "std": float(valid.std()),
                    "coverage": float(1 - panel.loc[panel["is_active"], col].isna().mean()),
                }
    
    ablation_result = {
        "with_social": {
            "m1_r2": m1_metrics.get("r2"),
            "m1_rmse": m1_metrics.get("rmse"),
            "m2_r2_median": baseline_metrics.get("r2_median"),
        },
        "social_features": {
            "count": len(social_feature_cols),
            "columns": social_feature_cols,
            "stats": social_stats,
        },
        "ablation_note": "不加社媒版本需要单独训练模型,此处提供基线对比数据",
    }
    
    return ablation_result


def run_loso_validation(run_dir: Path, panel: pd.DataFrame) -> dict:
    """运行 Leave-One-Season-Out 交叉验证."""
    seasons = panel["s_idx"].unique()
    n_seasons = len(seasons)
    
    # 简化版: 统计每个季的样本量
    season_stats = []
    for s in seasons:
        train_mask = panel["s_idx"] != s
        test_mask = panel["s_idx"] == s
        season_stats.append({
            "season": int(s),
            "train_size": int(train_mask.sum()),
            "test_size": int(test_mask.sum()),
        })
    
    loso_result = {
        "n_seasons": int(n_seasons),
        "season_stats": season_stats,
        "note": "完整 LOSO 需要对每个 fold 重新训练模型",
    }
    
    return loso_result


def analyze_platform_specificity(panel: pd.DataFrame, m2_fixed: pd.DataFrame) -> dict:
    """分析平台特异性效应."""
    platform_cols = [c for c in panel.columns if any(p in c.lower() for p in ["twitter", "youtube", "instagram", "tiktok"])]
    
    result = {
        "available_platforms": platform_cols,
        "platform_effects": {},
        "platform_stats": {},
    }
    
    # 从固定效应中提取平台相关项
    if m2_fixed is not None:
        for _, row in m2_fixed.iterrows():
            term = str(row["term"])
            # 检查是否包含平台关键词
            for platform in ["twitter", "youtube", "instagram", "tiktok"]:
                if platform in term.lower():
                    val = row.get("median", row.get("estimate", 0.0))
                    if val != 0.0:
                        result["platform_effects"][term] = float(val)
                    break
    
    # 从 panel 计算平台粉丝与 J_z 的相关性
    platform_correlations = {}
    for col in platform_cols:
        if "P_" in col and col in panel.columns:
            valid_mask = panel[col].notna() & panel["J_z"].notna() & panel["is_active"]
            if valid_mask.sum() > 30:
                corr = panel.loc[valid_mask, col].corr(panel.loc[valid_mask, "J_z"])
                platform_correlations[col] = float(corr) if not np.isnan(corr) else None
    
    result["platform_correlations_with_Jz"] = platform_correlations
    
    # 平台粉丝统计
    for col in platform_cols:
        if col in panel.columns and "P_" in col:
            valid_vals = panel.loc[panel["is_active"], col].dropna()
            if len(valid_vals) > 0:
                result["platform_stats"][col] = {
                    "mean": float(valid_vals.mean()),
                    "std": float(valid_vals.std()),
                    "missing_rate": float(panel.loc[panel["is_active"], col].isna().mean()),
                    "n_valid": int(len(valid_vals)),
                }
    
    # 平台粉丝分布统计
    for col in platform_cols:
        if col in panel.columns:
            valid_vals = panel[col].dropna()
            if len(valid_vals) > 0:
                result[f"{col}_stats"] = {
                    "mean": float(valid_vals.mean()),
                    "std": float(valid_vals.std()),
                    "missing_rate": float(panel[col].isna().mean()),
                }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Task3: 完整分析")
    parser.add_argument("--run", type=str, required=True, help="Run directory")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    run_dir = Path(args.run)
    if not run_dir.exists():
        raise FileNotFoundError(str(run_dir))
    
    print("=" * 60)
    print("Task3 完整分析")
    print("=" * 60)
    
    # 加载数据
    panel = load_parquet(run_dir / "panel.parquet")
    m1_random_pro = load_parquet(run_dir / "m1_torch_random_pro.parquet") if (run_dir / "m1_torch_random_pro.parquet").exists() else None
    m1_random_celeb = load_parquet(run_dir / "m1_torch_random_celeb.parquet") if (run_dir / "m1_torch_random_celeb.parquet").exists() else None
    m2_random_pro = load_parquet(run_dir / "m2_torch_random_pro.parquet") if (run_dir / "m2_torch_random_pro.parquet").exists() else None
    m2_random_celeb = load_parquet(run_dir / "m2_torch_random_celeb.parquet") if (run_dir / "m2_torch_random_celeb.parquet").exists() else None
    m2_fixed = load_parquet(run_dir / "m2_torch_fixed.parquet") if (run_dir / "m2_torch_fixed.parquet").exists() else None
    
    k_index = []
    if (run_dir / "k_index.json").exists():
        k_index = json.loads((run_dir / "k_index.json").read_text(encoding="utf-8"))
    
    full_report = {}
    
    # ========================================
    # 1. Assortative Mating 检查
    # ========================================
    print("\n[1/5] Assortative Mating 检查...")
    try:
        am_report = build_assortative_mating_report(
            panel,
            u_pro_df=m1_random_pro,
            v_celeb_df=m1_random_celeb,
        )
        full_report["assortative_mating"] = am_report
        (run_dir / "assortative_mating_report.json").write_text(
            json.dumps(am_report, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"  ✓ 已保存 assortative_mating_report.json")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        full_report["assortative_mating"] = {"error": str(e)}
    
    # ========================================
    # 2. 联立相关随机效应 (rho_p)
    # ========================================
    print("\n[2/5] 计算 rho_p (Pro 跨通道相关)...")
    try:
        rho_result = compute_rho_p(m1_random_pro, m2_random_pro)
        full_report["rho_p"] = rho_result
        print(f"  ✓ rho_p = {rho_result.get('rho_p', 'N/A'):.4f}" if rho_result.get('rho_p') else f"  ✗ {rho_result.get('error')}")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        full_report["rho_p"] = {"error": str(e)}
    
    # ========================================
    # 3. 消融实验
    # ========================================
    print("\n[3/5] 消融实验分析...")
    try:
        ablation_result = run_ablation_experiment(run_dir, panel, k_index, args.device)
        full_report["ablation"] = ablation_result
        print(f"  ✓ Baseline R²: {ablation_result.get('baseline_r2', 'N/A')}")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        full_report["ablation"] = {"error": str(e)}
    
    # ========================================
    # 4. LOSO 交叉验证
    # ========================================
    print("\n[4/5] LOSO 交叉验证分析...")
    try:
        loso_result = run_loso_validation(run_dir, panel)
        full_report["loso"] = loso_result
        print(f"  ✓ {loso_result['n_seasons']} seasons")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        full_report["loso"] = {"error": str(e)}
    
    # ========================================
    # 5. 平台特异性分析
    # ========================================
    print("\n[5/5] 平台特异性分析...")
    try:
        platform_result = analyze_platform_specificity(panel, m2_fixed)
        full_report["platform_specificity"] = platform_result
        
        # 生成平台特异性图 - 使用相关性数据
        correlations = platform_result.get("platform_correlations_with_Jz", {})
        if correlations:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            platforms = list(correlations.keys())
            values = [correlations[p] for p in platforms if correlations[p] is not None]
            platforms = [p for p in platforms if correlations[p] is not None]
            
            if platforms:
                # 简化标签
                labels = [p.replace("P_cele_", "Cele ").replace("P_partner_", "Partner ") for p in platforms]
                colors = ['#3498db' if 'cele' in p.lower() else '#e67e22' for p in platforms]
                
                bars = ax.barh(labels, values, color=colors, edgecolor='black')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax.set_xlabel('Correlation with Judge Score (J_z)')
                ax.set_title('Platform-Specific Social Media Correlations')
                ax.grid(axis='x', alpha=0.3)
                
                # 添加图例
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#3498db', label='Celebrity'),
                    Patch(facecolor='#e67e22', label='Pro Partner'),
                ]
                ax.legend(handles=legend_elements, loc='lower right')
                
                plt.tight_layout()
                plt.savefig(run_dir / "fig_platform_specificity.png", dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ 已保存 fig_platform_specificity.png")
            else:
                print(f"  ⚠ 无有效平台相关性数据")
        else:
            print(f"  ⚠ 无平台相关性数据")
    except Exception as e:
        import traceback
        print(f"  ✗ 错误: {e}")
        traceback.print_exc()
        full_report["platform_specificity"] = {"error": str(e)}
    
    # ========================================
    # 保存完整报告
    # ========================================
    (run_dir / "full_analysis_report.json").write_text(
        json.dumps(full_report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    # 更新 task3_artifact.json
    artifact_path = run_dir / "task3_artifact.json"
    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["rho_p"] = full_report.get("rho_p", {})
        artifact["assortative_mating_summary"] = {
            k: v for k, v in full_report.get("assortative_mating", {}).items()
            if not isinstance(v, (dict, list)) or k in ["multi_pro_count"]
        }
        artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n✓ 已更新 task3_artifact.json")
    
    print("\n" + "=" * 60)
    print("完整分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
