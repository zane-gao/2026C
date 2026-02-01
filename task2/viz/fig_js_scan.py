from __future__ import annotations

from pathlib import Path
import numpy as np


def plot(out_path: str | Path, data=None) -> None:
    """绘制 Judges Save 强度扫描图 (Fig.7)
    
    展示不同 JS 强度参数下的指标变化：
    - IR (Intercept Rate): 争议选手被救比例
    - Merit: 技术优胜度
    - Pop: 人气主权度
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        raise RuntimeError("matplotlib and seaborn required for plotting") from exc

    if data is None:
        print("[WARN] No data found for fig_js_scan")
        return
    
    traj_map = data.get("traj_map", {})
    report = data.get("report", {})
    cfg = data.get("cfg", {})
    
    # Try to extract JS-related metrics from report
    metrics_dict = report.get("metrics", {}).get("A", {})
    
    # Collect JS mechanism data
    js_data = {}
    for mech, vals in metrics_dict.items():
        if mech.startswith("JS") or mech == "JS":
            js_data[mech] = vals
    
    # Also look for IR in metrics
    ir_val = metrics_dict.get("IR", None)
    
    # If we have trajectory data with Bottom2/Save info, compute statistics
    js_stats = []
    for (mode, mech), traj in traj_map.items():
        if not mech.startswith("JS") and mech != "JS":
            continue
        
        bottom2 = traj.get("bottom2_mask")
        saved = traj.get("saved_mask")
        S_bar = traj.get("S_bar")
        F_bar = traj.get("F_bar")
        
        if bottom2 is not None and saved is not None:
            K, S, T, N = bottom2.shape
            # Compute save rate
            total_bottom2 = bottom2.sum()
            total_saved = (bottom2 & saved).sum()
            save_rate = float(total_saved / total_bottom2) if total_bottom2 > 0 else 0.0
            
            js_stats.append({
                "Mechanism": f"{mode}-{mech}",
                "Save Rate": save_rate,
                "Bottom2 Count": int(total_bottom2),
                "Saved Count": int(total_saved)
            })
    
    # Check if we have comparison data (P, R, JS)
    mechanism_metrics = []
    for mech in ["P", "R", "JS"]:
        if mech in metrics_dict:
            m = metrics_dict[mech]
            merit = m.get("Merit", {}).get("mean", float('nan'))
            pop = m.get("Pop", {}).get("mean", float('nan'))
            mechanism_metrics.append({
                "Mechanism": mech,
                "Merit": merit,
                "Pop": pop
            })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. JS Save Statistics
    if js_stats:
        import pandas as pd
        df_js = pd.DataFrame(js_stats)
        ax1 = axes[0]
        x = range(len(df_js))
        bars = ax1.bar(x, df_js["Save Rate"], color="steelblue", alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_js["Mechanism"], rotation=45, ha='right')
        ax1.set_title("Judges Save Statistics")
        ax1.set_ylabel("Save Rate (Proportion)")
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add count annotations (Saved / Bottom2)
        for i, (bar, row) in enumerate(zip(bars, df_js.itertuples())):
            saved_count = int(row._4)  # Saved Count
            bottom2_count = int(row._3)  # Bottom2 Count
            ax1.annotate(f'{saved_count}/{bottom2_count}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8)
    else:
        # Plot placeholder with IR if available
        ax1 = axes[0]
        if ir_val is not None and np.isfinite(ir_val):
            ax1.bar(["JS"], [ir_val], color="steelblue", alpha=0.7)
            ax1.set_ylabel("Intercept Rate (IR)")
            ax1.set_title("Judges Save: Controversy Intercept Rate")
        else:
            ax1.text(0.5, 0.5, "No JS save data available\n(Run JS mechanism simulation)", 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title("Judges Save Scan")
        ax1.grid(axis='y', alpha=0.3)
    
    # 2. Mechanism Comparison (Merit vs Pop)
    ax2 = axes[1]
    if mechanism_metrics:
        import pandas as pd
        df_mech = pd.DataFrame(mechanism_metrics)
        x = np.arange(len(df_mech))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, df_mech["Merit"], width, label='Merit', color='forestgreen', alpha=0.7)
        bars2 = ax2.bar(x + width/2, df_mech["Pop"], width, label='Pop', color='royalblue', alpha=0.7)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(df_mech["Mechanism"])
        ax2.set_title("Merit vs Pop by Mechanism")
        ax2.set_ylabel("Correlation Score")
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No mechanism comparison data available", 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Mechanism Comparison")
    
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
