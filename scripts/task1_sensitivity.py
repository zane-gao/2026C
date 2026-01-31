"""
Task1 灵敏度分析脚本 (PyTorch 版)。

对超参数进行网格扫描，评估模型性能对参数选择的敏感性。
关键参数:
- epsilon: 约束松弛
- tau: softmax 温度
- kappa: percent rule 的温度
- lambda_ent: 熵正则化权重
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task1.config import load_config
from task1.types import RuleParams
from task1.eval.torch_eval import (
    sample_posterior,
    compute_posterior_summary,
    replay_elimination,
    compute_feasible_rate,
)


def prepare_data(cfg):
    """加载并预处理数据 (只需做一次)。"""
    from task1.io.load_csv import load_csv
    from task1.data.reshape import wide_to_long
    from task1.data.build_masks import build_masks
    from task1.data.features import build_features
    from task1.data.tensors import build_tensors, build_week_obs_list

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
    
    return tensors, week_obs


def train_model_fast(
    tensors,
    week_obs,
    cfg,
    params: RuleParams,
    device: torch.device,
    epochs: int = 2000,
    lr: float = 0.01,
    patience: int = 200,
) -> dict:
    """
    快速训练模型并返回结果。
    
    Returns:
        dict with best_elbo, final_elbo, and trained model
    """
    from task1.model.torch_model import build_torch_model
    
    model = build_torch_model(tensors, cfg, device=device)
    season_idx = {s: i for i, s in enumerate(tensors.seasons)}
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-5)
    
    best_elbo = float("-inf")
    best_state = None
    no_improve = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        elbo, log_lik, kl = model.elbo(week_obs, season_idx, params, n_samples=1)
        loss = -elbo
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()
        
        elbo_val = elbo.item()
        if elbo_val > best_elbo:
            best_elbo = elbo_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    # 恢复最佳状态
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return {
        "best_elbo": best_elbo,
        "final_epoch": epoch,
        "model": model,
    }


def evaluate_model(model, tensors, week_obs, params, n_samples: int = 100) -> dict:
    """评估模型，返回 PPC 指标。"""
    model.eval()
    
    # 采样并计算指标
    samples = sample_posterior(model, n_samples=n_samples)
    summary = compute_posterior_summary(samples)
    
    F_mean = summary["F_mean"]
    ppc_results = replay_elimination(F_mean, tensors, week_obs, params)
    feasible_results = compute_feasible_rate(samples, tensors, week_obs, params, epsilon=params.epsilon)
    
    return {
        "accuracy": ppc_results["metrics"]["accuracy"],
        "cover_at_2": ppc_results["metrics"]["cover_at_2"],
        "mean_feasible_rate": feasible_results["mean_feasible_rate"],
        "mean_CI_width_90": float(np.nanmean(summary["CI_width_90"][tensors.active_mask > 0])),
        "mean_entropy": float(np.nanmean(summary["entropy_mean"])),
    }


def main():
    parser = argparse.ArgumentParser(description="Task1 灵敏度分析 (PyTorch)")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--out", type=str, default=None, help="输出目录")
    parser.add_argument("--device", type=str, default=None, help="设备: cuda 或 cpu")
    parser.add_argument("--epochs", type=int, default=2000, help="每次实验的训练轮数")
    parser.add_argument("--samples", type=int, default=100, help="评估时的采样数")
    parser.add_argument("--dry-run", action="store_true", help="只打印实验配置，不运行")
    args = parser.parse_args()
    
    # 设备
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # 输出目录
    out_dir = Path(args.out or "outputs/task1/sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    cfg = load_config(args.config)
    
    # 定义参数网格 (Smart Grid for Balanced Sensitivity Analysis)
    # 策略: 
    # 1. 对核心假设(Entropy)做细致扫描
    # 2. 对模型结构参数(Tau, Epsilon)做边界测试
    # 3. 固定次要参数(Kappa)以节省时间
    grid = {
        "epsilon": [0.005, 0.01],       # [严格, 宽松]
        "tau": [0.1, 0.2, 0.3],         # [尖锐, 适中, 平滑]
        "kappa": [20],                  # 固定为经验最优值
        "lambda_ent": [0.0, 0.05, 0.1, 0.2], # [无正则 -> 强正则] 观察O奖特性的影响
    }
    
    # 生成所有组合
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = list(itertools.product(*values))
    
    print(f"\n灵敏度分析: 共 {len(combos)} 个参数组合")
    print(f"参数网格:")
    for k, v in grid.items():
        print(f"  {k}: {v}")
    
    if args.dry_run:
        print(f"\n[DRY-RUN] 将运行以下实验:")
        for i, combo in enumerate(combos):
            params_str = ", ".join([f"{k}={v}" for k, v in zip(keys, combo)])
            print(f"  {i+1}. {params_str}")
        return
    
    # 加载数据 (只需一次)
    print("\n加载数据...")
    tensors, week_obs = prepare_data(cfg)
    print(f"  Seasons: {len(tensors.seasons)}, Observations: {len([o for o in week_obs if o.valid])}")
    
    # 运行实验
    results = []
    total = len(combos)
    start_time = time.time()
    
    print(f"\n开始实验...")
    print("=" * 70)
    
    for i, combo in enumerate(combos):
        param_dict = dict(zip(keys, combo))
        
        # 创建 RuleParams
        params = RuleParams(
            epsilon=param_dict.get("epsilon", cfg.hyper.epsilon),
            delta=cfg.hyper.delta,
            tau=param_dict.get("tau", cfg.hyper.tau),
            kappa=param_dict.get("kappa", cfg.hyper.kappa),
            kappa_r=cfg.hyper.kappa_r,
            alpha=cfg.hyper.alpha,
            kappa_b=cfg.hyper.kappa_b,
            eta1=cfg.hyper.eta1,
            eta2=cfg.hyper.eta2,
            lambda_ent=param_dict.get("lambda_ent", cfg.hyper.lambda_ent),
            bottom2_base=cfg.rules.bottom2_base,
            loglik_mode=cfg.rules.loglik_mode,
        )
        
        param_str = ", ".join([f"{k}={v}" for k, v in param_dict.items()])
        print(f"\n[{i+1}/{total}] {param_str}")
        
        # 训练
        exp_start = time.time()
        train_result = train_model_fast(
            tensors, week_obs, cfg, params, device,
            epochs=args.epochs, patience=200
        )
        
        # 评估
        eval_result = evaluate_model(
            train_result["model"], tensors, week_obs, params,
            n_samples=args.samples
        )
        
        exp_time = time.time() - exp_start
        
        # 记录结果
        result = {
            **param_dict,
            "best_elbo": train_result["best_elbo"],
            "final_epoch": train_result["final_epoch"],
            **eval_result,
            "time_sec": exp_time,
        }
        results.append(result)
        
        print(f"  ELBO={train_result['best_elbo']:.2f}, "
              f"Acc={eval_result['accuracy']:.1%}, "
              f"Cover@2={eval_result['cover_at_2']:.1%}, "
              f"Feasible={eval_result['mean_feasible_rate']:.1%}, "
              f"Time={exp_time:.0f}s")
        
        # 定期保存中间结果
        if (i + 1) % 5 == 0:
            pd.DataFrame(results).to_csv(out_dir / "sensitivity_results_partial.csv", index=False)
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"实验完成! 总时间: {total_time/60:.1f} 分钟")
    
    # 保存完整结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "sensitivity_results.csv", index=False)
    print(f"\n结果保存到: {out_dir / 'sensitivity_results.csv'}")
    
    # 分析结果
    print("\n最佳参数组合 (按 Accuracy):")
    best_acc = results_df.loc[results_df["accuracy"].idxmax()]
    for k in keys:
        print(f"  {k}: {best_acc[k]}")
    print(f"  Accuracy: {best_acc['accuracy']:.1%}")
    print(f"  Cover@2: {best_acc['cover_at_2']:.1%}")
    
    print("\n最佳参数组合 (按 ELBO):")
    best_elbo = results_df.loc[results_df["best_elbo"].idxmax()]
    for k in keys:
        print(f"  {k}: {best_elbo[k]}")
    print(f"  ELBO: {best_elbo['best_elbo']:.2f}")
    
    # 保存分析报告
    report = {
        "total_experiments": len(combos),
        "total_time_minutes": total_time / 60,
        "parameter_grid": grid,
        "best_by_accuracy": {
            "params": {k: float(best_acc[k]) for k in keys},
            "accuracy": float(best_acc['accuracy']),
            "cover_at_2": float(best_acc['cover_at_2']),
            "elbo": float(best_acc['best_elbo']),
        },
        "best_by_elbo": {
            "params": {k: float(best_elbo[k]) for k in keys},
            "accuracy": float(best_elbo['accuracy']),
            "cover_at_2": float(best_elbo['cover_at_2']),
            "elbo": float(best_elbo['best_elbo']),
        },
    }
    
    with open(out_dir / "sensitivity_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 灵敏度分析完成!")


if __name__ == "__main__":
    main()
