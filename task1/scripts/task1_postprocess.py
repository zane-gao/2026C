"""
Task1 后处理脚本：从训练好的模型生成所有交付物。

交付物：
1. F_hat (粉丝份额) 每个 (season, week, couple) 的后验均值 + 90%/95% CI
2. Certainty heatmap (CI 宽度)
3. PPC 指标: Acc, Cover@2, margin, feasible_rate  
4. 参数摘要 (beta 森林图数据)
5. 导出给 Task2/3/4/5 的接口数据
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# allow running from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task1.config import load_config
from task1.types import RuleParams, TensorPack, WeekObs
from task1.eval.torch_eval import (
    sample_posterior,
    compute_posterior_summary,
    replay_elimination,
    compute_feasible_rate,
    generate_fan_share_table,
)
from task1.viz.plots import (
    plot_certainty_heatmap,
    plot_entropy_heatmap,
    plot_forest_plot,
    plot_training_history,
    plot_ppc_accuracy_by_season,
)


def load_data_and_model(config_path: str, model_path: str, device: torch.device):
    """加载数据和训练好的模型。"""
    from task1.io.load_csv import load_csv
    from task1.data.reshape import wide_to_long
    from task1.data.build_masks import build_masks
    from task1.data.features import build_features
    from task1.data.tensors import build_tensors, build_week_obs_list
    from task1.model.torch_model import build_torch_model

    cfg = load_config(config_path)
    
    # Load data
    df = load_csv(cfg.paths.data_csv)
    long_df = wide_to_long(df, max_week=cfg.data.max_week)
    long_df, valid_df = build_masks(
        long_df,
        withdrawal_policy=cfg.data.withdrawal_policy,
        multi_elim_policy=cfg.data.multi_elim_policy,
    )
    feats_df, feature_cols = build_features(long_df)
    tensors = build_tensors(
        long_df, valid_df, feats_df, feature_cols,
        cfg.rules.season_rules, cfg.data.max_week
    )
    week_obs = build_week_obs_list(tensors)
    
    # Build model and load weights
    model = build_torch_model(tensors, cfg, device=device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Build rule params
    params = RuleParams(
        epsilon=cfg.hyper.epsilon,
        delta=cfg.hyper.delta,
        tau=cfg.hyper.tau,
        kappa=cfg.hyper.kappa,
        kappa_r=cfg.hyper.kappa_r,
        alpha=cfg.hyper.alpha,
        kappa_b=cfg.hyper.kappa_b,
        eta1=cfg.hyper.eta1,
        eta2=cfg.hyper.eta2,
        lambda_ent=cfg.hyper.lambda_ent,
        bottom2_base=cfg.rules.bottom2_base,
        loglik_mode=cfg.rules.loglik_mode,
    )
    
    return cfg, tensors, week_obs, model, params


def generate_all_outputs(
    tensors: TensorPack,
    week_obs: list,
    model,
    params: RuleParams,
    out_dir: Path,
    n_samples: int = 500,
):
    """生成所有 Task1 交付物。"""
    print(f"=" * 60)
    print("Task1 后处理: 生成所有交付物")
    print(f"=" * 60)
    
    # 1. 采样后验分布
    print(f"\n[1/6] 从后验分布采样 {n_samples} 次...")
    samples = sample_posterior(model, n_samples=n_samples)
    print(f"  -> theta 形状: {samples['theta'].shape}")
    print(f"  -> F 形状: {samples['F'].shape}")
    
    # 2. 计算后验统计量
    print("\n[2/6] 计算后验统计量 (均值, 中位数, CI)...")
    summary = compute_posterior_summary(samples)
    
    # 保存原始 numpy 数组
    np.savez(
        out_dir / "posterior_summary.npz",
        F_mean=summary["F_mean"],
        F_median=summary["F_median"],
        F_q05=summary["F_q05"],
        F_q95=summary["F_q95"],
        F_q025=summary["F_q025"],
        F_q975=summary["F_q975"],
        F_std=summary["F_std"],
        CI_width_90=summary["CI_width_90"],
        CI_width_95=summary["CI_width_95"],
        entropy_mean=summary["entropy_mean"],
        entropy_std=summary["entropy_std"],
    )
    print(f"  -> 保存到 posterior_summary.npz")
    
    # 3. 生成粉丝份额表 (主要交付物)
    print("\n[3/6] 生成粉丝份额表...")
    fan_share_rows = generate_fan_share_table(summary, tensors, week_obs)
    fan_share_df = pd.DataFrame(fan_share_rows)
    fan_share_df.to_csv(out_dir / "fan_share_estimates.csv", index=False)
    print(f"  -> 保存到 fan_share_estimates.csv ({len(fan_share_df)} 行)")
    
    # 4. PPC: 重放淘汰并计算指标
    print("\n[4/6] PPC: 重放淘汰...")
    F_mean = summary["F_mean"]
    ppc_results = replay_elimination(F_mean, tensors, week_obs, params)
    
    ppc_metrics = ppc_results["metrics"]
    print(f"  -> Accuracy: {ppc_metrics['accuracy']:.3f}")
    print(f"  -> Cover@2: {ppc_metrics['cover_at_2']:.3f}")
    print(f"  -> Total valid weeks: {ppc_metrics['total_valid_weeks']}")
    
    # 保存 PPC 详细结果
    ppc_details_df = pd.DataFrame(ppc_results["details"])
    ppc_details_df.to_csv(out_dir / "ppc_details.csv", index=False)
    
    with open(out_dir / "ppc_metrics.json", "w", encoding="utf-8") as f:
        json.dump(ppc_metrics, f, indent=2)
    print(f"  -> 保存到 ppc_metrics.json, ppc_details.csv")
    
    # 5. 可行率计算
    print("\n[5/6] 计算约束可行率...")
    feasible_results = compute_feasible_rate(samples, tensors, week_obs, params, epsilon=params.epsilon)
    print(f"  -> Mean feasible rate: {feasible_results['mean_feasible_rate']:.3f}")
    print(f"  -> Min feasible rate: {feasible_results['min_feasible_rate']:.3f}")
    
    with open(out_dir / "feasibility_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "mean_feasible_rate": float(feasible_results['mean_feasible_rate']),
            "min_feasible_rate": float(feasible_results['min_feasible_rate']),
        }, f, indent=2)
    
    # 6. 提取并保存模型参数
    print("\n[6/8] 提取模型参数...")
    beta_mu = model.beta_mu.detach().cpu().numpy()
    beta_std = np.exp(model.beta_logstd.detach().cpu().numpy())
    gamma_mu = model.gamma_mu.detach().cpu().numpy()
    gamma_std = np.exp(model.gamma_logstd.detach().cpu().numpy())
    
    # 特征重要性表
    feature_importance = []
    for i, name in enumerate(tensors.feature_names):
        feature_importance.append({
            "feature": name,
            "beta_mean": float(beta_mu[i]),
            "beta_std": float(beta_std[i]),
            "beta_lower": float(beta_mu[i] - 1.96 * beta_std[i]),
            "beta_upper": float(beta_mu[i] + 1.96 * beta_std[i]),
            "significant": bool(abs(beta_mu[i]) > 1.96 * beta_std[i]),
        })
    
    feature_df = pd.DataFrame(feature_importance)
    feature_df = feature_df.sort_values("beta_mean", key=abs, ascending=False)
    feature_df.to_csv(out_dir / "feature_importance.csv", index=False)
    print(f"  -> 保存到 feature_importance.csv")
    
    # 7. 生成可视化图表
    print("\n[7/8] 生成可视化图表...")
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # CI 宽度热力图
        plot_certainty_heatmap(
            summary["CI_width_90"],
            tensors.seasons,
            plot_dir / "certainty_heatmap.png",
        )
        print(f"  -> certainty_heatmap.png")
        
        # 熵热力图  
        plot_entropy_heatmap(
            summary["entropy_mean"],
            tensors.seasons,
            plot_dir / "entropy_heatmap.png",
        )
        print(f"  -> entropy_heatmap.png")
        
        # 特征系数森林图
        plot_forest_plot(
            tensors.feature_names,
            beta_mu,
            beta_std,
            plot_dir / "feature_forest_plot.png",
        )
        print(f"  -> feature_forest_plot.png")
        
        # PPC 准确率柱状图
        plot_ppc_accuracy_by_season(
            ppc_results["details"],
            tensors.seasons,
            plot_dir / "ppc_accuracy_by_season.png",
        )
        print(f"  -> ppc_accuracy_by_season.png")
        
    except Exception as e:
        print(f"  [WARN] 可视化生成失败: {e}")
    
    # 8. 汇总报告
    print("\n[8/8] 生成汇总报告...")
    print("=" * 60)
    
    report = {
        "model_info": {
            "n_seasons": len(tensors.seasons),
            "n_features": len(tensors.feature_names),
            "n_posterior_samples": n_samples,
        },
        "ppc_metrics": ppc_metrics,
        "feasibility": {
            "mean_feasible_rate": float(feasible_results['mean_feasible_rate']),
            "min_feasible_rate": float(feasible_results['min_feasible_rate']),
        },
        "uncertainty_summary": {
            "mean_CI_width_90": float(np.nanmean(summary["CI_width_90"][tensors.active_mask > 0])),
            "mean_CI_width_95": float(np.nanmean(summary["CI_width_95"][tensors.active_mask > 0])),
            "mean_entropy": float(np.nanmean(summary["entropy_mean"])),
        },
        "files_generated": [
            "fan_share_estimates.csv",
            "posterior_summary.npz",
            "ppc_metrics.json",
            "ppc_details.csv",
            "feasibility_metrics.json",
            "feature_importance.csv",
            "task1_report.json",
            "plots/certainty_heatmap.png",
            "plots/entropy_heatmap.png",
            "plots/feature_forest_plot.png",
            "plots/ppc_accuracy_by_season.png",
        ]
    }
    
    with open(out_dir / "task1_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n主要文件:")
    for f in report["files_generated"]:
        print(f"  - {f}")
    
    print(f"\nPPC 指标:")
    print(f"  Accuracy:  {ppc_metrics['accuracy']:.1%}")
    print(f"  Cover@2:   {ppc_metrics['cover_at_2']:.1%}")
    
    print(f"\n不确定性:")
    print(f"  Mean 90% CI width: {report['uncertainty_summary']['mean_CI_width_90']:.4f}")
    print(f"  Mean entropy:      {report['uncertainty_summary']['mean_entropy']:.4f}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Task1 后处理: 生成所有交付物")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--model", type=str, default=None, help="模型检查点路径 (默认: outputs/task1/run_torch/best_model.pt)")
    parser.add_argument("--out", type=str, default=None, help="输出目录 (默认: outputs/task1/results)")
    parser.add_argument("--samples", type=int, default=500, help="后验采样数 (默认: 500)")
    parser.add_argument("--device", type=str, default=None, help="设备: cuda 或 cpu")
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Default paths
    model_path = args.model or "outputs/task1/run_torch/best_model.pt"
    out_dir = Path(args.out or "outputs/task1/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Config: {args.config}")
    print(f"Model: {model_path}")
    print(f"Output: {out_dir}")
    print(f"Samples: {args.samples}")
    
    # Load
    cfg, tensors, week_obs, model, params = load_data_and_model(args.config, model_path, device)
    
    # Generate outputs
    report = generate_all_outputs(tensors, week_obs, model, params, out_dir, n_samples=args.samples)
    
    print("\n✅ Task1 后处理完成!")
    return report


if __name__ == "__main__":
    main()
