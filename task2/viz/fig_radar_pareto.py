from __future__ import annotations

from pathlib import Path
import numpy as np


def plot(out_path: str | Path, data=None) -> None:
    """绘制 Merit-Pop-Volatility 雷达图与 Pareto 前沿 (Fig.8)
    
    展示不同机制在三个核心指标上的权衡：
    - Merit: 技术优胜度 (最终名次与技能的相关性)
    - Pop: 人气主权度 (最终名次与观众份额的相关性)
    - Volatility: 戏剧性/波动性 (结果的不确定性)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        import seaborn as sns
    except Exception as exc:
        raise RuntimeError("matplotlib and seaborn required for plotting") from exc

    if data is None:
        print("[WARN] No data found for fig_radar_pareto")
        return
    
    report = data.get("report", {})
    traj_map = data.get("traj_map", {})
    
    # Extract metrics from report
    metrics_all = report.get("metrics", {})
    
    # Collect data for each mechanism
    mech_data = []
    for mode, mech_dict in metrics_all.items():
        if not isinstance(mech_dict, dict):
            continue
        for mech, vals in mech_dict.items():
            if not isinstance(vals, dict) or mech in ["consistency_P_R", "ATE_group", "ATE_group_exo", "ATE_group_quad", "IR"]:
                continue
            
            merit = vals.get("Merit", {})
            pop = vals.get("Pop", {})
            
            merit_val = merit.get("mean", float('nan')) if isinstance(merit, dict) else float('nan')
            pop_val = pop.get("mean", float('nan')) if isinstance(pop, dict) else float('nan')
            
            # Compute volatility from trajectory data
            vol_val = float('nan')
            key = (mode, mech)
            if key in traj_map:
                traj = traj_map[key]
                final_rank = traj.get("final_rank")  # [K, S, N]
                if final_rank is not None:
                    # Volatility = average std of ranks across simulations
                    K, S, N = final_rank.shape
                    vols = []
                    for s in range(S):
                        mask = final_rank[0, s] > 0
                        if mask.sum() > 0:
                            std_s = final_rank[:, s, mask].std(axis=0).mean()
                            vols.append(std_s)
                    if vols:
                        vol_val = float(np.mean(vols))
            
            # Try to get from report if not computed
            if np.isnan(vol_val):
                div = vals.get("Div", {})
                vol_val = div.get("mean", float('nan')) if isinstance(div, dict) else float('nan')
            
            if np.isfinite(merit_val) or np.isfinite(pop_val):
                mech_data.append({
                    "Mode": mode,
                    "Mechanism": mech,
                    "Label": f"{mode}-{mech}",
                    "Merit": merit_val,
                    "Pop": pop_val,
                    "Volatility": vol_val if np.isfinite(vol_val) else 0.5
                })
    
    if not mech_data:
        # Fallback: try to compute from trajectories directly
        for (mode, mech), traj in traj_map.items():
            S_bar = traj.get("S_bar")
            F_bar = traj.get("F_bar")
            final_rank = traj.get("final_rank")
            
            if S_bar is None or F_bar is None or final_rank is None:
                continue
            
            K, S, N = final_rank.shape
            
            # Compute Merit (correlation with skill)
            merit_vals = []
            pop_vals = []
            for k in range(min(K, 100)):  # Sample for speed
                for s in range(S):
                    rank = final_rank[k, s]
                    mask = rank > 0
                    if mask.sum() < 3:
                        continue
                    skill = S_bar[k, s, mask]
                    pop = F_bar[k, s, mask]
                    r = rank[mask]
                    
                    # Spearman correlation
                    from scipy.stats import spearmanr
                    try:
                        merit_vals.append(spearmanr(r, -skill)[0])
                        pop_vals.append(spearmanr(r, -pop)[0])
                    except:
                        pass
            
            merit_val = float(np.nanmean(merit_vals)) if merit_vals else float('nan')
            pop_val = float(np.nanmean(pop_vals)) if pop_vals else float('nan')
            
            # Volatility
            vols = []
            for s in range(S):
                mask = final_rank[0, s] > 0
                if mask.sum() > 0:
                    std_s = final_rank[:, s, mask].std(axis=0).mean()
                    vols.append(std_s)
            vol_val = float(np.mean(vols)) if vols else 0.5
            
            mech_data.append({
                "Mode": mode,
                "Mechanism": mech,
                "Label": f"{mode}-{mech}",
                "Merit": merit_val,
                "Pop": pop_val,
                "Volatility": vol_val
            })
    
    if not mech_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No mechanism metrics available", ha='center', va='center', 
                fontsize=14, transform=ax.transAxes)
        ax.set_title("Merit-Pop-Volatility Analysis")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return
    
    import pandas as pd
    df = pd.DataFrame(mech_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Scatter plot: Merit vs Pop (Pareto-like view)
    ax1 = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    
    # Add small jitter to avoid overlapping points
    np.random.seed(42)
    jitter_scale = 0.02
    
    for i, row in df.iterrows():
        if np.isfinite(row["Merit"]) and np.isfinite(row["Pop"]):
            size = max(100, 300 * row["Volatility"]) if np.isfinite(row["Volatility"]) else 150
            # Add small jitter
            merit_jitter = row["Merit"] + np.random.uniform(-jitter_scale, jitter_scale)
            pop_jitter = row["Pop"] + np.random.uniform(-jitter_scale, jitter_scale)
            ax1.scatter(merit_jitter, pop_jitter, s=size, c=[colors[i]], 
                       label=row["Label"], alpha=0.7, edgecolors='black')
            # Add label next to point with offset to avoid overlap
            offset_x = 10 + i * 5
            offset_y = -10 - i * 8
            ax1.annotate(row["Label"], (merit_jitter, pop_jitter),
                        textcoords="offset points", xytext=(offset_x, offset_y),
                        fontsize=8, alpha=0.8)
    
    ax1.set_xlabel("Merit (Skill-Rank Correlation)")
    ax1.set_ylabel("Pop (Popularity-Rank Correlation)")
    ax1.set_title("Merit vs Pop Trade-off\n(Size ∝ Volatility)")
    ax1.legend(loc='upper left', fontsize=8, bbox_to_anchor=(0.02, 0.98))
    ax1.grid(True, alpha=0.3)
    
    # Add ideal point marker
    ax1.scatter([1], [1], marker='*', s=200, c='gold', edgecolors='black', zorder=10)
    
    # 2. Bar chart comparison
    ax2 = axes[1]
    x = np.arange(len(df))
    width = 0.25
    
    merit_vals = df["Merit"].fillna(0).values
    pop_vals = df["Pop"].fillna(0).values
    vol_vals = df["Volatility"].fillna(0).values
    
    bars1 = ax2.bar(x - width, merit_vals, width, label='Merit', color='forestgreen', alpha=0.7)
    bars2 = ax2.bar(x, pop_vals, width, label='Pop', color='royalblue', alpha=0.7)
    bars3 = ax2.bar(x + width, vol_vals, width, label='Volatility', color='coral', alpha=0.7)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["Label"], rotation=45, ha='right')
    ax2.set_ylabel("Score")
    ax2.set_title("Mechanism Comparison")
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
