from __future__ import annotations

from pathlib import Path
import numpy as np

def plot(out_path: str | Path, data=None) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except Exception as exc:
        raise RuntimeError("matplotlib/seaborn required for plotting") from exc

    if data is None or "report" not in data:
        print("[WARN] No report data found for fig_rank_bar")
        return

    metrics_dict = data["report"].get("metrics", {}).get("A", {})
    if not metrics_dict:
        return

    # Prepare DataFrame for plotting
    records = []
    mechs = ["P", "R", "JS"]
    # Available metrics to compare
    target_metrics = ["Merit", "Pop", "Regret_S", "Regret_F"]
    
    for m in mechs:
        if m not in metrics_dict:
            continue
        vals = metrics_dict[m]
        for metric in target_metrics:
            if metric in vals:
                # We use the mean value
                val = vals[metric].get("mean", 0.0)
                # Also get error bars if possible (q_low, q_high) - but barplot handles this better with raw data
                # Here we only have summary stats in report.json
                records.append({
                    "Mechanism": m,
                    "Metric": metric,
                    "Value": val
                })
    
    if not records:
        return

    df = pd.DataFrame(records)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Merit vs Pop
    sub_df1 = df[df["Metric"].isin(["Merit", "Pop"])]
    sns.barplot(data=sub_df1, x="Metric", y="Value", hue="Mechanism", ax=axes[0], palette="viridis")
    axes[0].set_title("Meritocracy (Tech) vs Audience Sovereignty (Pop)")
    axes[0].set_ylabel("Correlation / Score")
    axes[0].grid(axis='y', alpha=0.3)

    # 2. Regret (Lower is better)
    sub_df2 = df[df["Metric"].isin(["Regret_S", "Regret_F"])]
    sns.barplot(data=sub_df2, x="Metric", y="Value", hue="Mechanism", ax=axes[1], palette="magma")
    axes[1].set_title("Regret (Lower is Better)")
    axes[1].set_ylabel("Regret Value")
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
