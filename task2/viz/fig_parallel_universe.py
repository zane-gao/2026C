from __future__ import annotations

from pathlib import Path
import numpy as np

def plot(out_path: str | Path, data=None) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        raise RuntimeError("matplotlib and seaborn required for plotting") from exc

    if data is None or "traj_map" not in data:
        print("[WARN] No trajectory data found for fig_parallel_universe")
        return

    traj_map = data["traj_map"]
    # Prefer Rank or JS mode to show variance
    keys = list(traj_map.keys())
    target_key = None
    for k in [("A", "JS"), ("A", "R"), ("A", "P")]:
        if k in keys:
            target_key = k
            break
    
    if target_key is None:
        target_key = keys[0]

    traj = traj_map[target_key]
    final_rank = traj["final_rank"] # Shape [K, S, N]
    K, S, N = final_rank.shape

    # Select a season with interesting variance
    # We look for season with max std dev in rankings
    std_per_season = final_rank.std(axis=0).mean() # [S] -> scalar (mean over contestants)
    # Actually calculate scalar variability score per season
    var_scores = []
    for s in range(S):
        # average std deviation across contestants who participated
        mask = final_rank[0, s] > 0
        if mask.any():
            std_s = final_rank[:, s, mask].std(axis=0).mean()
            var_scores.append(std_s)
        else:
            var_scores.append(0)
    
    # Pick top variability season
    best_s = np.argmax(var_scores)
    
    # Prepare data for boxplot
    season_ranks = final_rank[:, best_s, :] # [K, N]
    
    # Filter contestants who actually played
    mask_c = season_ranks[0] > 0
    contestant_indices = np.where(mask_c)[0]
    
    # Sort contestants by median rank
    median_ranks = np.median(season_ranks[:, contestant_indices], axis=0)
    sorted_idx = contestant_indices[np.argsort(median_ranks)]

    # Flatten for seaborn
    plot_data = []
    for cid in sorted_idx:
        ranks = season_ranks[:, cid]
        for r in ranks:
            plot_data.append({
                "Contestant": f"C{cid+1}", 
                "Rank": r
            })
    
    if not plot_data:
        return

    import pandas as pd
    df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="Contestant", y="Rank", hue="Contestant", ax=ax, palette="coolwarm", legend=False)
    ax.set_title(f"Parallel Universes: Rank Variability in Season {best_s+1}\n(Mechanism: {target_key[1]})")
    ax.invert_yaxis() # Rank 1 at top
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
