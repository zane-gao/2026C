from __future__ import annotations

from pathlib import Path
import numpy as np


def plot(out_path: str | Path, data=None) -> None:
    """绘制权重敏感性分析图 (Fig.10)
    
    展示混合机制 (W_MIX) 在不同权重 w 下的表现：
    - w=0: 纯 Rank 机制
    - w=1: 纯 Percent 机制
    - 中间值: 混合机制
    
    分析指标随权重变化的趋势与 Pareto 前沿
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        raise RuntimeError("matplotlib and seaborn required for plotting") from exc

    if data is None:
        print("[WARN] No data found for fig_weight_sensitivity")
        return
    
    report = data.get("report", {})
    traj_map = data.get("traj_map", {})
    cfg = data.get("cfg", {})
    
    # Look for weight scan data in trajectories (W_MIX variants)
    w_scan_data = []
    
    # Check for W_MIX mechanisms with different weights
    for (mode, mech), traj in traj_map.items():
        if "W_MIX" in mech or "w_" in mech.lower():
            # Try to extract weight from mechanism name
            try:
                if "_" in mech:
                    w_str = mech.split("_")[-1]
                    w = float(w_str)
                else:
                    w = 0.5  # default
            except:
                w = 0.5
            
            # Compute metrics for this weight
            final_rank = traj.get("final_rank")
            S_bar = traj.get("S_bar")
            F_bar = traj.get("F_bar")
            
            if final_rank is not None and S_bar is not None and F_bar is not None:
                K, S, N = final_rank.shape
                merit_vals = []
                pop_vals = []
                
                for k in range(min(K, 50)):
                    for s in range(S):
                        rank = final_rank[k, s]
                        mask = rank > 0
                        if mask.sum() < 3:
                            continue
                        from scipy.stats import spearmanr
                        try:
                            merit_vals.append(spearmanr(rank[mask], -S_bar[k, s, mask])[0])
                            pop_vals.append(spearmanr(rank[mask], -F_bar[k, s, mask])[0])
                        except:
                            pass
                
                if merit_vals and pop_vals:
                    w_scan_data.append({
                        "Weight (w)": w,
                        "Merit": float(np.nanmean(merit_vals)),
                        "Pop": float(np.nanmean(pop_vals)),
                        "Mechanism": f"{mode}-{mech}"
                    })
    
    # If no W_MIX data, use P and R as endpoints
    metrics_all = report.get("metrics", {})
    for mode, mech_dict in metrics_all.items():
        if not isinstance(mech_dict, dict):
            continue
        for mech, vals in mech_dict.items():
            if mech == "P":
                merit = vals.get("Merit", {}).get("mean", float('nan'))
                pop = vals.get("Pop", {}).get("mean", float('nan'))
                w_scan_data.append({"Weight (w)": 1.0, "Merit": merit, "Pop": pop, "Mechanism": f"{mode}-P"})
            elif mech == "R":
                merit = vals.get("Merit", {}).get("mean", float('nan'))
                pop = vals.get("Pop", {}).get("mean", float('nan'))
                w_scan_data.append({"Weight (w)": 0.0, "Merit": merit, "Pop": pop, "Mechanism": f"{mode}-R"})
            elif mech == "JS":
                merit = vals.get("Merit", {}).get("mean", float('nan'))
                pop = vals.get("Pop", {}).get("mean", float('nan'))
                w_scan_data.append({"Weight (w)": 0.5, "Merit": merit, "Pop": pop, "Mechanism": f"{mode}-JS"})
    
    if not w_scan_data:
        # Create synthetic sensitivity demonstration
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate theoretical curve
        w_grid = np.linspace(0, 1, 11)
        merit_theo = 0.5 + 0.3 * w_grid + 0.1 * np.random.randn(len(w_grid)) * 0.1
        pop_theo = 0.8 - 0.2 * w_grid + 0.1 * np.random.randn(len(w_grid)) * 0.1
        
        ax.plot(w_grid, merit_theo, 'o-', label='Merit', color='forestgreen', markersize=8)
        ax.plot(w_grid, pop_theo, 's-', label='Pop', color='royalblue', markersize=8)
        ax.fill_between(w_grid, merit_theo - 0.05, merit_theo + 0.05, alpha=0.2, color='forestgreen')
        ax.fill_between(w_grid, pop_theo - 0.05, pop_theo + 0.05, alpha=0.2, color='royalblue')
        
        ax.set_xlabel("Weight w (0=Rank, 1=Percent)")
        ax.set_ylabel("Correlation Score")
        ax.set_title("Weight Sensitivity Analysis\n(Theoretical Demonstration)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        
        ax.text(0.5, 0.15, "No W_MIX scan data available.\nRun simulation with mixture mechanisms for actual results.",
               ha='center', va='center', transform=ax.transAxes, fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return
    
    import pandas as pd
    df = pd.DataFrame(w_scan_data)
    df = df.sort_values("Weight (w)")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Metrics vs Weight
    ax1 = axes[0]
    
    # Group by weight and plot
    unique_weights = sorted(df["Weight (w)"].unique())
    
    if len(unique_weights) > 1:
        # Plot as lines if we have multiple weights
        merit_by_w = df.groupby("Weight (w)")["Merit"].mean()
        pop_by_w = df.groupby("Weight (w)")["Pop"].mean()
        
        ax1.plot(merit_by_w.index, merit_by_w.values, 'o-', label='Merit', 
                color='forestgreen', markersize=10, linewidth=2)
        ax1.plot(pop_by_w.index, pop_by_w.values, 's-', label='Pop', 
                color='royalblue', markersize=10, linewidth=2)
    else:
        # Plot as scatter with labels
        for _, row in df.iterrows():
            ax1.scatter(row["Weight (w)"], row["Merit"], c='forestgreen', s=100, 
                       label='Merit' if _ == 0 else '', marker='o')
            ax1.scatter(row["Weight (w)"], row["Pop"], c='royalblue', s=100, 
                       label='Pop' if _ == 0 else '', marker='s')
            ax1.annotate(row["Mechanism"], (row["Weight (w)"], row["Merit"]), 
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    ax1.set_xlabel("Weight w (0=Rank, 1=Percent)")
    ax1.set_ylabel("Correlation Score")
    ax1.set_title("Metrics vs Mechanism Weight")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.1, 1.1)
    
    # 2. Pareto Front (Merit vs Pop scatter)
    ax2 = axes[1]
    
    colors = plt.cm.coolwarm(df["Weight (w)"].values)
    scatter = ax2.scatter(df["Merit"], df["Pop"], c=df["Weight (w)"], 
                         cmap='coolwarm', s=150, alpha=0.8, edgecolors='black')
    
    # Add labels with smart positioning to avoid overlap
    label_offsets = {}  # Track used positions
    for idx, row in df.iterrows():
        # Use different offsets based on index to reduce overlap
        base_offset_x = 8
        base_offset_y = 8
        # Stagger offsets for different points
        if idx % 3 == 0:
            offset_x, offset_y = base_offset_x, base_offset_y
        elif idx % 3 == 1:
            offset_x, offset_y = base_offset_x + 15, base_offset_y - 15
        else:
            offset_x, offset_y = base_offset_x - 10, base_offset_y + 15
        
        ax2.annotate(row["Mechanism"], (row["Merit"], row["Pop"]), 
                    textcoords="offset points", xytext=(offset_x, offset_y), 
                    fontsize=8, ha='left')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Weight (w)")
    
    ax2.set_xlabel("Merit (Skill-Rank Correlation)")
    ax2.set_ylabel("Pop (Popularity-Rank Correlation)")
    ax2.set_title("Trade-off Space (Pareto View)")
    ax2.grid(True, alpha=0.3)
    
    # Mark ideal point
    ax2.scatter([1], [1], marker='*', s=300, c='gold', edgecolors='black', zorder=10)
    ax2.annotate("Ideal", (1, 1), textcoords="offset points", xytext=(-15, -15), fontsize=9)
    
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
