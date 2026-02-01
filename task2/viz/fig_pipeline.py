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
        return
        
    traj_map = data["traj_map"]
    # Plot Score Distributions (J vs F) for one mechanism
    # Just to show the input data distribution
    
    target_key = list(traj_map.keys())[0]
    traj = traj_map[target_key]
    
    # We have F_bar [K, S, N] and maybe inferred J?
    # Actually simulate.py used J internally but might not have saved raw J distribution in trajectory
    # But we can plot F_bar distribution
    
    F_bar = traj["F_bar"].flatten()
    S_bar = traj["S_bar"].flatten()
    
    # Filter valid
    mask = (F_bar > 0) & (S_bar != 0)
    F_vals = F_bar[mask]
    S_vals = S_bar[mask]
    
    if len(F_vals) > 1000:
        idx = np.random.choice(len(F_vals), 1000, replace=False)
        F_vals = F_vals[idx]
        S_vals = S_vals[idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=S_vals, y=F_vals, alpha=0.3, ax=ax)
    ax.set_title("Input Latent Structure: Skill vs Popularity (Sampled)")
    ax.set_xlabel("Latent Skill (S)")
    ax.set_ylabel("Audience Share (F)")
    ax.grid(True, alpha=0.3)
    
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
