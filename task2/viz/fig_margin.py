from __future__ import annotations

from pathlib import Path
import numpy as np


def plot(out_path: str | Path, data=None) -> None:
    """绘制不同机制下的 margin 分布对比图 (Fig.5)
    
    Margin 定义：被淘汰选手与"安全线"选手之间的得分差异
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        raise RuntimeError("matplotlib and seaborn required for plotting") from exc

    if data is None or "traj_map" not in data:
        print("[WARN] No trajectory data found for fig_margin")
        return

    traj_map = data["traj_map"]
    
    # 收集不同机制的 margin 数据
    margin_data = {}
    for (mode, mech), traj in traj_map.items():
        if "margin" not in traj:
            continue
        margin = traj["margin"]  # shape [K, S, T] or [K, S, T, N]
        
        # Flatten and filter valid values
        if margin.ndim == 4:
            # Per-contestant margins: take mean over contestants
            margin_flat = margin.mean(axis=-1).flatten()
        else:
            margin_flat = margin.flatten()
        
        valid = margin_flat[np.isfinite(margin_flat) & (margin_flat != 0)]
        if valid.size > 0:
            key = f"{mode}-{mech}"
            margin_data[key] = valid
    
    if not margin_data:
        # Create a placeholder with message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No margin data available", ha='center', va='center', 
                fontsize=14, transform=ax.transAxes)
        ax.set_title("Margin Distribution")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return
    
    # Prepare data for plotting
    plot_records = []
    for key, vals in margin_data.items():
        # Sample if too many points
        if len(vals) > 5000:
            vals = np.random.choice(vals, 5000, replace=False)
        for v in vals:
            plot_records.append({"Mechanism": key, "Margin": v})
    
    import pandas as pd
    df = pd.DataFrame(plot_records)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Histogram/KDE by mechanism
    for key in margin_data.keys():
        subset = df[df["Mechanism"] == key]["Margin"]
        sns.kdeplot(subset, ax=axes[0], label=key, fill=True, alpha=0.3)
    axes[0].set_title("Margin Distribution by Mechanism (KDE)")
    axes[0].set_xlabel("Margin (Score Difference from Safety)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    sns.boxplot(data=df, x="Mechanism", y="Margin", hue="Mechanism", ax=axes[1], palette="Set2", legend=False)
    axes[1].set_title("Margin Comparison Across Mechanisms")
    axes[1].set_xlabel("Mechanism")
    axes[1].set_ylabel("Margin")
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
