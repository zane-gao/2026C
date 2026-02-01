from __future__ import annotations

from pathlib import Path
import numpy as np


def plot(out_path: str | Path, data=None) -> None:
    """绘制偏向性箱线图 (Fig.6)
    
    展示 ITE (Individual Treatment Effect) 的分布：
    - tau_T: 生存时间效应 (在不同机制下的存活周数差异)
    - tau_pi: 最终名次效应 (在不同机制下的最终排名差异)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        raise RuntimeError("matplotlib and seaborn required for plotting") from exc

    if data is None or "ite_rows" not in data:
        print("[WARN] No ITE data found for fig_bias_box")
        # Try to extract from trajectory data
        if data is not None and "traj_map" in data:
            # Compute simple bias from trajectories
            _plot_from_traj(out_path, data["traj_map"])
            return
        return
    
    ite_rows = data["ite_rows"]
    if not ite_rows:
        print("[WARN] Empty ITE rows for fig_bias_box")
        return
    
    # Prepare data for plotting
    plot_records = []
    for row in ite_rows:
        tau_T = row.get("tau_T_mean", float('nan'))
        tau_pi = row.get("tau_pi_mean", float('nan'))
        in_controversy = row.get("in_controversy", False)
        in_skill_top = row.get("in_skill_top", False)
        in_pop_top = row.get("in_pop_top", False)
        
        if np.isfinite(tau_T):
            group = "Controversy" if in_controversy else "Non-Controversy"
            plot_records.append({"Effect": "Survival (τ_T)", "Value": tau_T, "Group": group})
        if np.isfinite(tau_pi):
            group = "Controversy" if in_controversy else "Non-Controversy"
            plot_records.append({"Effect": "Rank (τ_π)", "Value": -tau_pi, "Group": group})  # Negative for "benefit"
    
    if not plot_records:
        print("[WARN] No valid ITE values for fig_bias_box")
        return
    
    import pandas as pd
    df = pd.DataFrame(plot_records)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Survival Effect by Group
    df_T = df[df["Effect"] == "Survival (τ_T)"]
    if not df_T.empty:
        sns.boxplot(data=df_T, x="Group", y="Value", hue="Group", ax=axes[0], palette={"Controversy": "salmon", "Non-Controversy": "lightblue"}, legend=False)
        sns.stripplot(data=df_T, x="Group", y="Value", ax=axes[0], alpha=0.3, size=3, color="black")
        axes[0].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[0].set_title("Survival Effect (τ_T)\n(Positive = Survived Longer in Alternative)")
        axes[0].set_xlabel("Contestant Group")
        axes[0].set_ylabel("Weeks Gained/Lost")
        axes[0].grid(axis='y', alpha=0.3)
    
    # 2. Rank Effect by Group
    df_pi = df[df["Effect"] == "Rank (τ_π)"]
    if not df_pi.empty:
        sns.boxplot(data=df_pi, x="Group", y="Value", hue="Group", ax=axes[1], palette={"Controversy": "salmon", "Non-Controversy": "lightblue"}, legend=False)
        sns.stripplot(data=df_pi, x="Group", y="Value", ax=axes[1], alpha=0.3, size=3, color="black")
        axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[1].set_title("Rank Effect (-τ_π)\n(Positive = Better Final Rank in Alternative)")
        axes[1].set_xlabel("Contestant Group")
        axes[1].set_ylabel("Rank Improvement")
        axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_from_traj(out_path, traj_map):
    """Fallback: compute simple bias from trajectory data"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
    except Exception:
        return
    
    # Compare P vs R if available
    key_p = ("A", "P")
    key_r = ("A", "R")
    
    if key_p not in traj_map or key_r not in traj_map:
        keys = list(traj_map.keys())
        if len(keys) < 2:
            return
        key_p, key_r = keys[0], keys[1]
    
    traj_p = traj_map[key_p]
    traj_r = traj_map[key_r]
    
    rank_p = traj_p["final_rank"]  # [K, S, N]
    rank_r = traj_r["final_rank"]
    
    K, S, N = rank_p.shape
    
    # Compute rank difference per contestant
    diff_records = []
    for s in range(S):
        for i in range(N):
            rp = rank_p[:, s, i]
            rr = rank_r[:, s, i]
            valid = (rp > 0) & (rr > 0)
            if valid.sum() > 10:
                diff = (rr[valid] - rp[valid]).mean()  # Positive = better in P
                diff_records.append({"Season": s + 1, "Contestant": i, "Rank Diff (R-P)": diff})
    
    if not diff_records:
        return
    
    df = pd.DataFrame(diff_records)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Rank Diff (R-P)"], kde=True, ax=ax, color="steelblue")
    ax.axvline(0, color='r', linestyle='--', alpha=0.7)
    ax.set_title(f"Rank Bias: {key_r} vs {key_p}\n(Positive = Better Rank in P)")
    ax.set_xlabel("Mean Rank Difference")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
