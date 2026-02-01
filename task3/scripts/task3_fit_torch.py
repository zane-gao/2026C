"""
Task3 PyTorch 训练脚本.
使用 CUDA 加速拟合 M1/M2/M3 模型.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task3.config import load_config, Config
from task3.io.export import load_parquet, save_parquet, save_json
from task3.io.dataset import build_dataset, find_data_csv
from task3.io.task1_artifact import load_task1_artifact
from task3.data.features import prepare_features
from task3.data.tensors import (
    panel_to_tensors,
    build_y_matrix_torch,
    get_device,
    TensorPack,
)
from task3.models.torch_m1_judges import fit_m1_torch, extract_m1_results
from task3.models.torch_m2_fans import fit_m2_torch, extract_m2_results
from task3.models.torch_m3_survival import fit_m3_torch, extract_m3_results


def _load_k_index(path: Path) -> list[int]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def print_device_info(device: torch.device) -> None:
    """打印设备信息."""
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        print(f"  CUDA version: {torch.version.cuda}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task3: fit models with PyTorch (CUDA accelerated)")
    parser.add_argument("--run", type=str, required=True, help="Run directory (outputs/task3/run_xxx)")
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda, cpu, or auto")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=500, help="Max epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--skip-m1", action="store_true", help="Skip M1 model")
    parser.add_argument("--skip-m2", action="store_true", help="Skip M2 model")
    parser.add_argument("--skip-m3", action="store_true", help="Skip M3 model")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        raise FileNotFoundError(str(run_dir))

    # 加载配置
    cfg_path = run_dir / "config.resolved.yaml"
    cfg = load_config(str(cfg_path)) if cfg_path.exists() else load_config(None)

    # 设备
    device = get_device(args.device)
    print_device_info(device)
    print()

    # 加载 panel
    print("Loading panel data...")
    panel = load_parquet(run_dir / "panel.parquet")
    panel, feature_spec = prepare_features(
        panel,
        include_social=cfg.features.include_social,
        include_platform=cfg.features.include_platform,
        missing_as_zero=cfg.features.missing_as_zero,
        winsorize_q=cfg.features.winsorize_q,
    )
    print(f"  Panel shape: {panel.shape}")

    # 转换为张量
    print("Converting to tensors...")
    tensor_pack = panel_to_tensors(
        panel,
        feature_spec,
        response_col="J_z",
        week_effect=cfg.features.week_effect,
        device=args.device,
    )
    print(f"  Features: {len(tensor_pack.feature_names)}")
    print(f"  n_pro: {tensor_pack.n_pro}, n_celeb: {tensor_pack.n_celeb}, n_season: {tensor_pack.n_season}")

    # 加载 task1 artifact (for y_matrix)
    print("Loading Task1 artifact...")
    data_csv = find_data_csv(cfg.paths.data_csv)
    dataset = build_dataset(data_csv, max_week=cfg.panel.max_week)
    artifact = load_task1_artifact(cfg.paths.task1_artifact, dataset=dataset, max_week=cfg.panel.max_week)
    
    k_index = _load_k_index(run_dir / "k_index.json")
    if not k_index:
        k_index = list(range(min(artifact.theta.shape[0], cfg.runtime.k_subsample)))
    print(f"  K samples: {len(k_index)}")

    # 计算 y_matrix for M2
    print("Building Y matrix...")
    panel_m2 = panel.copy()
    panel_m2 = panel_m2[panel_m2["is_active"]]
    if cfg.panel.use_valid_mask:
        panel_m2 = panel_m2[panel_m2["is_valid_week"]]
    panel_m2 = panel_m2[np.isfinite(panel_m2["J_z"])]
    panel_m2 = panel_m2.reset_index(drop=True)

    s_idx = panel_m2["s_idx"].to_numpy()
    t_idx = panel_m2["t_idx"].to_numpy()
    i_idx = panel_m2["i_idx"].to_numpy()
    
    S = artifact.theta.shape[1]
    T = artifact.theta.shape[2]
    ref_idx = np.zeros((S, T), dtype=int)
    for (s, t), grp in panel.groupby(["s_idx", "t_idx"]):
        ref_idx[int(s), int(t)] = int(grp["ref_couple_id"].iloc[0])

    # 构建 y_matrix (numpy)
    theta_np = artifact.theta
    y_matrix_np = np.zeros((len(k_index), len(s_idx)), dtype=np.float32)
    for kk, k in enumerate(k_index):
        ref = ref_idx[s_idx, t_idx]
        y_matrix_np[kk] = theta_np[k, s_idx, t_idx, i_idx] - theta_np[k, s_idx, t_idx, ref]
    
    y_matrix = torch.tensor(y_matrix_np, dtype=torch.float32, device=device)
    print(f"  Y matrix shape: {y_matrix.shape}")
    print()

    use_amp = not args.no_amp and device.type == "cuda"
    results = {}

    # ==================== M1 ====================
    if not args.skip_m1:
        print("=" * 60)
        print("Fitting M1 (Judge scores model) with PyTorch...")
        print("=" * 60)
        t0 = time.time()
        
        m1_model, m1_history = fit_m1_torch(
            tensor_pack,
            lr=args.lr,
            max_epochs=args.epochs,
            patience=args.patience,
            verbose=True,
            use_amp=use_amp,
        )
        
        m1_results = extract_m1_results(
            m1_model,
            tensor_pack,
            panel,
            tensor_pack.feature_names,
        )
        
        t1 = time.time()
        print(f"M1 completed in {t1 - t0:.1f}s")
        print(f"  R2: {m1_results['metrics']['r2']:.4f}")
        print(f"  RMSE: {m1_results['metrics']['rmse']:.4f}")
        
        # 保存结果
        save_parquet(run_dir / "m1_torch_fixed.parquet", m1_results["fixed"])
        save_parquet(run_dir / "m1_torch_random_pro.parquet", m1_results["random_pro"])
        save_parquet(run_dir / "m1_torch_random_celeb.parquet", m1_results["random_celeb"])
        save_parquet(run_dir / "m1_torch_random_season.parquet", m1_results["random_season"])
        save_json(run_dir / "m1_torch_metrics.json", m1_results["metrics"])
        save_json(run_dir / "m1_torch_var_components.json", m1_results["var_components"])
        
        results["m1"] = m1_results["metrics"]
        print()

    # ==================== M2 ====================
    if not args.skip_m2:
        print("=" * 60)
        print("Fitting M2 (Fan share model) with PyTorch...")
        print("=" * 60)
        t0 = time.time()
        
        # 为 M2 创建新的 tensor_pack
        tensor_pack_m2 = panel_to_tensors(
            panel_m2,
            feature_spec,
            response_col="J_z",
            week_effect=cfg.features.week_effect,
            device=args.device,
        )
        
        m2_model, m2_history = fit_m2_torch(
            tensor_pack_m2,
            y_matrix,
            k_index,
            lr=args.lr,
            max_epochs=args.epochs,
            patience=args.patience,
            verbose=True,
            use_amp=use_amp,
        )
        
        m2_results = extract_m2_results(
            m2_model,
            tensor_pack_m2,
            y_matrix,
            k_index,
            panel_m2,
            tensor_pack_m2.feature_names,
        )
        
        t1 = time.time()
        print(f"M2 completed in {t1 - t0:.1f}s")
        print(f"  R2 median: {m2_results['metrics']['r2_median']:.4f}")
        
        # 保存结果
        save_parquet(run_dir / "m2_torch_fixed.parquet", m2_results["fixed_summary"])
        save_parquet(run_dir / "m2_torch_random_pro.parquet", m2_results["random_pro_summary"])
        save_parquet(run_dir / "m2_torch_random_celeb.parquet", m2_results["random_celeb_summary"])
        save_json(run_dir / "m2_torch_metrics.json", m2_results["metrics"])
        save_json(run_dir / "m2_torch_var_components.json", m2_results["var_components"])
        
        results["m2"] = m2_results["metrics"]
        print()

    # ==================== M3 ====================
    if not args.skip_m3:
        print("=" * 60)
        print("Fitting M3 (Survival model) with PyTorch...")
        print("=" * 60)
        t0 = time.time()
        
        m3_model, m3_history = fit_m3_torch(
            tensor_pack_m2 if not args.skip_m2 else tensor_pack,
            y_matrix,
            k_index,
            lr=args.lr * 0.5,  # 通常 logistic 需要更小的学习率
            max_epochs=args.epochs,
            patience=args.patience,
            verbose=True,
            use_amp=use_amp,
        )
        
        m3_results = extract_m3_results(
            m3_model,
            tensor_pack_m2 if not args.skip_m2 else tensor_pack,
            y_matrix,
            tensor_pack_m2.feature_names if not args.skip_m2 else tensor_pack.feature_names,
        )
        
        t1 = time.time()
        print(f"M3 completed in {t1 - t0:.1f}s")
        print(f"  AUC: {m3_results['metrics']['auc']:.4f}")
        print(f"  Brier: {m3_results['metrics']['brier']:.4f}")
        print(f"  eta_J: {m3_results['metrics']['eta_J']:.4f}")
        print(f"  eta_F: {m3_results['metrics']['eta_F']:.4f}")
        
        # 保存结果
        save_parquet(run_dir / "m3_torch_fixed.parquet", m3_results["fixed"])
        save_json(run_dir / "m3_torch_metrics.json", m3_results["metrics"])
        
        results["m3"] = m3_results["metrics"]
        print()

    # 保存汇总
    save_json(run_dir / "torch_results_summary.json", results)
    
    print("=" * 60)
    print("All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
