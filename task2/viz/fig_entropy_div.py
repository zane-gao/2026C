from __future__ import annotations

from pathlib import Path
import numpy as np


def plot(out_path: str | Path, data=None) -> None:
    """绘制熵与发散度分析图 (Fig.9)
    
    展示不同机制的结果不确定性：
    - H_win: 冠军分布的熵 (越高表示结果越不确定)
    - Divergence: 与现实结果的差异程度
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        raise RuntimeError("matplotlib and seaborn required for plotting") from exc

    if data is None:
        print("[WARN] No data found for fig_entropy_div")
        return
    
    report = data.get("report", {})
    traj_map = data.get("traj_map", {})
    
    # Collect entropy and divergence data
    mech_entropy = {}
    mech_div = {}
    
    # From report
    metrics_all = report.get("metrics", {})
    for mode, mech_dict in metrics_all.items():
        if not isinstance(mech_dict, dict):
            continue
        for mech, vals in mech_dict.items():
            if not isinstance(vals, dict) or mech in ["consistency_P_R", "ATE_group", "ATE_group_exo", "ATE_group_quad", "IR"]:
                continue
            
            label = f"{mode}-{mech}"
            
            # Get H_win (entropy of winner distribution)
            h_win = vals.get("H_win", {})
            if isinstance(h_win, dict):
                mech_entropy[label] = h_win.get("mean", float('nan'))
            
            # Get Divergence
            div = vals.get("Div", {})
            if isinstance(div, dict):
                mech_div[label] = div.get("mean", float('nan'))
    
    # Compute from trajectories if not available
    for (mode, mech), traj in traj_map.items():
        label = f"{mode}-{mech}"
        
        # Compute winner entropy
        if label not in mech_entropy or np.isnan(mech_entropy.get(label, float('nan'))):
            winner = traj.get("winner")  # [K, S]
            if winner is not None:
                K, S = winner.shape
                entropies = []
                for s in range(S):
                    w = winner[:, s]
                    valid = w[w >= 0]
                    if valid.size > 10:
                        # Compute entropy of winner distribution
                        counts = np.bincount(valid.astype(int))
                        probs = counts[counts > 0] / counts.sum()
                        entropy = -np.sum(probs * np.log2(probs + 1e-10))
                        entropies.append(entropy)
                if entropies:
                    mech_entropy[label] = float(np.mean(entropies))
        
        # Compute divergence (std of final ranks as proxy)
        if label not in mech_div or np.isnan(mech_div.get(label, float('nan'))):
            final_rank = traj.get("final_rank")  # [K, S, N]
            if final_rank is not None:
                K, S, N = final_rank.shape
                divs = []
                for s in range(S):
                    mask = final_rank[0, s] > 0
                    if mask.sum() > 0:
                        # Coefficient of variation of ranks
                        std_s = final_rank[:, s, mask].std(axis=0).mean()
                        mean_s = final_rank[:, s, mask].mean()
                        if mean_s > 0:
                            divs.append(std_s / mean_s)
                if divs:
                    mech_div[label] = float(np.mean(divs))
    
    if not mech_entropy and not mech_div:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No entropy/divergence data available", ha='center', va='center', 
                fontsize=14, transform=ax.transAxes)
        ax.set_title("Entropy and Divergence Analysis")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Winner Entropy by Mechanism
    ax1 = axes[0]
    if mech_entropy:
        labels = list(mech_entropy.keys())
        values = [mech_entropy[k] for k in labels]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
        
        bars = ax1.bar(range(len(labels)), values, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel("Winner Entropy (bits)")
        ax1.set_title("H_win: Uncertainty of Champion\n(Higher = More Uncertain)")
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if np.isfinite(val):
                ax1.annotate(f'{val:.2f}',
                            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            ha='center', va='bottom', fontsize=9)
    else:
        ax1.text(0.5, 0.5, "No entropy data", ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("Winner Entropy")
    
    # 2. Divergence by Mechanism
    ax2 = axes[1]
    if mech_div:
        labels = list(mech_div.keys())
        values = [mech_div[k] for k in labels]
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(labels)))
        
        bars = ax2.bar(range(len(labels)), values, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel("Divergence (CV of Ranks)")
        ax2.set_title("Outcome Divergence\n(Higher = More Variable Results)")
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if np.isfinite(val):
                ax2.annotate(f'{val:.3f}',
                            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No divergence data", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Divergence")
    
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
