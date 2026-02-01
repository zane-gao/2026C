from __future__ import annotations

from pathlib import Path
import numpy as np

def plot(out_path: str | Path, data=None) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        raise RuntimeError("matplotlib and seaborn required for plotting") from exc

    if data is None or "ite_rows" not in data:
        print("[WARN] No ITE data found for fig_survival_ite_ate")
        return

    ite_rows = data["ite_rows"]
    if not ite_rows:
        return

    # Extract data for plotting
    tau_T = [row["tau_T_mean"] for row in ite_rows]
    tau_pi = [row["tau_pi_mean"] for row in ite_rows]
    in_controversy = [row["in_controversy"] for row in ite_rows]
    season = [row["season"] for row in ite_rows]

    # Convert to numpy for easier masking
    tau_T = np.array(tau_T)
    tau_pi = np.array(tau_pi)
    in_controversy = np.array(in_controversy)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Distribution of ITE on Survival Time (Rank vs Percent)
    # Positive tau_T means survived longer in Percent (since tau = T_P - T_R usually)
    # Check definition in compute_ite: tau = T_target - T_control. 
    # Usually we do P - R or JS - R. Assume the compute_ite done P - R.
    
    from matplotlib.ticker import MaxNLocator
    sns.histplot(tau_T, kde=False, ax=axes[0], color='skyblue', edgecolor='black', discrete=True, stat="percent")
    axes[0].set_title("Distribution of Survival Effect (ITE)\n(Positive = Survived Longer in Percent vs Rank)")
    axes[0].set_xlabel("Change in Survival Weeks (Weeks)")
    axes[0].set_ylabel("Percentage of Contestants (%)")
    axes[0].axvline(0, color='r', linestyle='--', alpha=0.5)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 2. Scatter: Rank Change vs Survival Change, colored by Controversy
    # tau_pi negative means better rank (smaller number)
    sns.scatterplot(x=tau_T, y=-tau_pi, hue=in_controversy, palette={True: "red", False: "gray"}, 
                    alpha=0.6, ax=axes[1], s=40)
    axes[1].set_title("Survival Gain vs Rank Gain\n(Red: Controversy Group)")
    axes[1].set_xlabel("Survival Benefit (Weeks)")
    axes[1].set_ylabel("Rank Benefit (Negative of Rank Change)")
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].legend(title="Controversy")

    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
