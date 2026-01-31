"""
Task1 可视化模块：生成 heatmap, forest plot 等图表。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def _ensure_matplotlib():
    """确保 matplotlib 可用。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        return plt
    except ImportError as e:
        raise RuntimeError("matplotlib required for plotting. Run: pip install matplotlib") from e


def plot_certainty_heatmap(
    CI_width: np.ndarray,  # [S, T, N] or [S, T]
    seasons: List[int],
    out_path: Path,
    title: str = "Fan Share Uncertainty (90% CI Width)",
    figsize: tuple = (14, 10),
    cmap: str = "YlOrRd",
) -> Path:
    """
    绘制不确定性热力图 (按 season-week 平均)。
    
    Args:
        CI_width: CI 宽度数组
        seasons: 赛季列表
        out_path: 输出文件路径
        title: 图标题
    
    Returns:
        保存的图片路径
    """
    plt = _ensure_matplotlib()
    
    # 如果是 3D，沿 N 维度平均
    if CI_width.ndim == 3:
        # 只对活跃选手取平均
        ci_mean = np.nanmean(CI_width, axis=-1)  # [S, T]
    else:
        ci_mean = CI_width
    
    S, T = ci_mean.shape
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(ci_mean, aspect='auto', cmap=cmap)
    
    # 设置标签
    ax.set_yticks(range(S))
    ax.set_yticklabels([f"S{s}" for s in seasons], fontsize=8)
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"W{w+1}" for w in range(T)], fontsize=8)
    
    ax.set_xlabel("Week", fontsize=12)
    ax.set_ylabel("Season", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("CI Width", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return out_path


def plot_entropy_heatmap(
    entropy: np.ndarray,  # [S, T]
    seasons: List[int],
    out_path: Path,
    title: str = "Fan Share Distribution Entropy",
    figsize: tuple = (14, 10),
    cmap: str = "viridis",
) -> Path:
    """
    绘制熵热力图。
    
    熵越高表示粉丝份额分布越均匀（不确定性越大）。
    """
    plt = _ensure_matplotlib()
    
    S, T = entropy.shape
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(entropy, aspect='auto', cmap=cmap)
    
    ax.set_yticks(range(S))
    ax.set_yticklabels([f"S{s}" for s in seasons], fontsize=8)
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"W{w+1}" for w in range(T)], fontsize=8)
    
    ax.set_xlabel("Week", fontsize=12)
    ax.set_ylabel("Season", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Entropy (nats)", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return out_path


def plot_forest_plot(
    feature_names: List[str],
    beta_mean: np.ndarray,
    beta_std: np.ndarray,
    out_path: Path,
    title: str = "Feature Coefficients (β)",
    figsize: tuple = (10, 8),
    ci_level: float = 1.96,
) -> Path:
    """
    绘制特征系数森林图。
    
    显示每个特征的 β 均值和 95% CI。
    """
    plt = _ensure_matplotlib()
    
    n_features = len(feature_names)
    
    # 按绝对值排序
    order = np.argsort(np.abs(beta_mean))[::-1]
    names_sorted = [feature_names[i] for i in order]
    means_sorted = beta_mean[order]
    stds_sorted = beta_std[order]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(n_features)
    errors = ci_level * stds_sorted
    
    # 不显著的用灰色
    colors = []
    for mean, err in zip(means_sorted, errors):
        if mean - err > 0 or mean + err < 0:
            colors.append('#2E86AB')  # 显著 (蓝色)
        else:
            colors.append('#AAAAAA')  # 不显著 (灰色)
    
    ax.barh(y_pos, means_sorted, xerr=errors, align='center',
            color=colors, alpha=0.7, capsize=3)
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.set_xlabel("Coefficient Value", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    ax.invert_yaxis()  # 最重要的在顶部
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return out_path


def plot_training_history(
    history_path: Path,
    out_path: Path,
    figsize: tuple = (12, 8),
) -> Path:
    """
    绘制训练历史曲线。
    """
    plt = _ensure_matplotlib()
    
    data = np.load(history_path)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # ELBO
    ax = axes[0, 0]
    ax.plot(data['elbo'], alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO")
    ax.set_title("ELBO")
    ax.grid(True, alpha=0.3)
    
    # Log-likelihood
    ax = axes[0, 1]
    ax.plot(data['log_lik'], alpha=0.7, color='green')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log-likelihood")
    ax.set_title("Log-likelihood")
    ax.grid(True, alpha=0.3)
    
    # KL divergence
    ax = axes[1, 0]
    ax.plot(data['kl'], alpha=0.7, color='red')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence")
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 1]
    ax.plot(data['lr'], alpha=0.7, color='purple')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Training History", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return out_path


def plot_ppc_accuracy_by_season(
    ppc_details: List[Dict],
    seasons: List[int],
    out_path: Path,
    figsize: tuple = (12, 6),
) -> Path:
    """
    按赛季绘制 PPC 准确率柱状图。
    """
    plt = _ensure_matplotlib()
    
    # 按赛季统计
    from collections import defaultdict
    season_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for row in ppc_details:
        s = row["season"]
        season_stats[s]["total"] += 1
        if row["correct"]:
            season_stats[s]["correct"] += 1
    
    sorted_seasons = sorted(season_stats.keys())
    accuracies = [season_stats[s]["correct"] / season_stats[s]["total"] 
                  if season_stats[s]["total"] > 0 else 0 
                  for s in sorted_seasons]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = range(len(sorted_seasons))
    bars = ax.bar(x_pos, accuracies, color='steelblue', alpha=0.7)
    
    # 标注数值
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.0%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"S{s}" for s in sorted_seasons], fontsize=8, rotation=45)
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("PPC Elimination Accuracy by Season", fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=np.mean(accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(accuracies):.1%}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return out_path


def plot_fan_share_evolution(
    F_mean: np.ndarray,  # [S, T, N]
    season_idx: int,
    couple_names: List[str],
    active_mask: np.ndarray,  # [S, T, N]
    out_path: Path,
    figsize: tuple = (12, 6),
) -> Path:
    """
    绘制某一赛季的粉丝份额随时间演变。
    """
    plt = _ensure_matplotlib()
    
    F_season = F_mean[season_idx]  # [T, N]
    mask_season = active_mask[season_idx]  # [T, N]
    T, N = F_season.shape
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i in range(N):
        # 只画有效的周
        valid_weeks = np.where(mask_season[:, i] > 0)[0]
        if len(valid_weeks) > 0:
            name = couple_names[i] if i < len(couple_names) else f"Couple {i}"
            ax.plot(valid_weeks + 1, F_season[valid_weeks, i], 
                    marker='o', markersize=4, label=name, alpha=0.7)
    
    ax.set_xlabel("Week", fontsize=12)
    ax.set_ylabel("Fan Share", fontsize=12)
    ax.set_title(f"Fan Share Evolution", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return out_path


def generate_all_plots(
    summary: Dict[str, np.ndarray],
    tensors,
    ppc_details: List[Dict],
    history_path: Path,
    out_dir: Path,
) -> List[Path]:
    """
    生成所有可视化图表。
    
    Returns:
        生成的图表文件路径列表
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = []
    
    print("生成可视化图表...")
    
    # 1. CI 宽度热力图
    try:
        path = plot_certainty_heatmap(
            summary["CI_width_90"],
            tensors.seasons,
            out_dir / "certainty_heatmap.png",
        )
        plots.append(path)
        print(f"  -> {path.name}")
    except Exception as e:
        print(f"  [WARN] certainty_heatmap failed: {e}")
    
    # 2. 熵热力图
    try:
        path = plot_entropy_heatmap(
            summary["entropy_mean"],
            tensors.seasons,
            out_dir / "entropy_heatmap.png",
        )
        plots.append(path)
        print(f"  -> {path.name}")
    except Exception as e:
        print(f"  [WARN] entropy_heatmap failed: {e}")
    
    # 3. 训练历史
    try:
        path = plot_training_history(
            history_path,
            out_dir / "training_history.png",
        )
        plots.append(path)
        print(f"  -> {path.name}")
    except Exception as e:
        print(f"  [WARN] training_history failed: {e}")
    
    # 4. PPC 准确率柱状图
    try:
        path = plot_ppc_accuracy_by_season(
            ppc_details,
            tensors.seasons,
            out_dir / "ppc_accuracy_by_season.png",
        )
        plots.append(path)
        print(f"  -> {path.name}")
    except Exception as e:
        print(f"  [WARN] ppc_accuracy_by_season failed: {e}")
    
    return plots
