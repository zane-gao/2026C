"""
Visualization module for Task1 PyTorch model.
Generates certainty heatmaps, forest plots, and other diagnostic visualizations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Defer matplotlib import to avoid issues in non-GUI environments
def _import_matplotlib():
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    return plt, mcolors


def plot_certainty_heatmap(
    summary: Dict[str, np.ndarray],
    seasons: List[int],
    save_path: Optional[Path] = None,
    metric: str = "CI_width_90",
    title: str = "Fan Share Uncertainty Heatmap",
    figsize: Tuple[int, int] = (16, 10),
) -> None:
    """
    Plot certainty/uncertainty heatmap across seasons and weeks.
    
    Args:
        summary: Posterior summary dict with CI_width_90, entropy_mean, etc.
        seasons: List of season numbers
        save_path: Path to save figure
        metric: Which metric to plot ("CI_width_90", "CI_width_95", "entropy_mean")
        title: Plot title
        figsize: Figure size
    """
    plt, mcolors = _import_matplotlib()
    
    # Get metric data
    if metric in summary:
        data = summary[metric]
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Aggregate across couples (take mean per (s, t))
    if data.ndim == 3:  # [S, T, N] -> [S, T]
        # Use mean of active couples
        with np.errstate(invalid='ignore'):
            data_agg = np.nanmean(np.where(data > 0, data, np.nan), axis=-1)
    else:
        data_agg = data  # Already [S, T]
    
    S, T = data_agg.shape
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(data_agg, aspect='auto', cmap='RdYlGn_r')  # Red = high uncertainty
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric, fontsize=12)
    
    # Axis labels
    ax.set_xlabel("Week", fontsize=12)
    ax.set_ylabel("Season", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Ticks
    ax.set_xticks(np.arange(T))
    ax.set_xticklabels([f"W{i+1}" for i in range(T)], fontsize=8)
    ax.set_yticks(np.arange(S))
    ax.set_yticklabels([f"S{s}" for s in seasons], fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    
    plt.close(fig)


def plot_entropy_heatmap(
    summary: Dict[str, np.ndarray],
    seasons: List[int],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 10),
) -> None:
    """
    Plot entropy heatmap (higher = more uncertainty about who will be eliminated).
    """
    plot_certainty_heatmap(
        summary=summary,
        seasons=seasons,
        save_path=save_path,
        metric="entropy_mean",
        title="Posterior Entropy Heatmap (Higher = More Uncertainty)",
        figsize=figsize,
    )


def plot_fan_share_forest(
    summary: Dict[str, np.ndarray],
    season_idx: int,
    week_idx: int,
    couple_names: List[str],
    active_mask: np.ndarray,
    elim_idx: Optional[int] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Forest plot showing fan share estimates with confidence intervals for one week.
    
    Args:
        summary: Posterior summary dict
        season_idx: Index into seasons array
        week_idx: Index into weeks (0-based)
        couple_names: List of couple names
        active_mask: [N] boolean array of active couples
        elim_idx: Index of eliminated couple (for highlighting)
        save_path: Path to save figure
    """
    plt, mcolors = _import_matplotlib()
    
    F_mean = summary["F_mean"][season_idx, week_idx]  # [N]
    F_q05 = summary["F_q05"][season_idx, week_idx]
    F_q95 = summary["F_q95"][season_idx, week_idx]
    F_q025 = summary["F_q025"][season_idx, week_idx]
    F_q975 = summary["F_q975"][season_idx, week_idx]
    
    # Filter to active couples
    active_idx = np.where(active_mask > 0)[0]
    n_active = len(active_idx)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by F_mean (highest to lowest)
    sorted_order = np.argsort(F_mean[active_idx])[::-1]
    plot_idx = active_idx[sorted_order]
    
    y_pos = np.arange(n_active)
    
    for i, idx in enumerate(plot_idx):
        color = 'red' if idx == elim_idx else 'steelblue'
        
        # 95% CI
        ax.plot([F_q025[idx], F_q975[idx]], [y_pos[i], y_pos[i]], 
                color=color, alpha=0.3, linewidth=8)
        
        # 90% CI
        ax.plot([F_q05[idx], F_q95[idx]], [y_pos[i], y_pos[i]], 
                color=color, alpha=0.6, linewidth=4)
        
        # Mean point
        ax.scatter([F_mean[idx]], [y_pos[i]], color=color, s=50, zorder=10)
    
    # Labels
    labels = [couple_names[idx] if idx < len(couple_names) else f"Couple {idx}" 
              for idx in plot_idx]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    
    ax.set_xlabel("Fan Share (F)", fontsize=12)
    ax.set_title(f"Fan Share Estimates with 90%/95% CI", fontsize=14)
    
    ax.axvline(x=1/n_active, color='gray', linestyle='--', alpha=0.5, 
               label=f'Uniform ({1/n_active:.3f})')
    ax.legend()
    
    ax.set_xlim(0, max(F_q975[active_idx]) * 1.1)
    ax.invert_yaxis()  # Highest at top
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved forest plot to {save_path}")
    
    plt.close(fig)


def plot_training_history(
    history_path: Path,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Plot training history (ELBO, KL, LogLik over epochs).
    """
    plt, _ = _import_matplotlib()
    
    history = np.load(history_path)
    epochs = history["epochs"]
    elbo = history["elbo"]
    kl = history["kl"]
    loglik = history["loglik"]
    lr = history["lr"]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # ELBO
    ax = axes[0, 0]
    ax.plot(epochs, elbo, color='steelblue')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO")
    ax.set_title("ELBO (Evidence Lower Bound)")
    ax.grid(True, alpha=0.3)
    
    # KL Divergence
    ax = axes[0, 1]
    ax.plot(epochs, kl, color='orange')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence (Regularization)")
    ax.grid(True, alpha=0.3)
    
    # Log-Likelihood
    ax = axes[1, 0]
    ax.plot(epochs, loglik, color='green')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title("Log-Likelihood (Data Fit)")
    ax.grid(True, alpha=0.3)
    
    # Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, lr, color='red')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    
    plt.close(fig)


def plot_ppc_summary(
    ppc_results: Dict,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot PPC (Posterior Predictive Check) summary.
    """
    plt, _ = _import_matplotlib()
    
    metrics = ppc_results["metrics"]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar chart of main metrics
    ax = axes[0]
    metric_names = ["Accuracy", "Cover@2"]
    metric_values = [metrics["accuracy"], metrics["cover_at_2"]]
    colors = ['steelblue', 'green']
    
    bars = ax.bar(metric_names, metric_values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("PPC Metrics")
    
    # Add value labels
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', fontsize=12)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.legend()
    
    # Margin distribution
    ax = axes[1]
    margins = [d["margin"] for d in ppc_results["details"] if d["correct"]]
    margins_wrong = [d["margin"] for d in ppc_results["details"] if not d["correct"]]
    
    if margins:
        ax.hist(margins, bins=20, alpha=0.7, label='Correct', color='green')
    if margins_wrong:
        ax.hist(margins_wrong, bins=20, alpha=0.7, label='Incorrect', color='red')
    
    ax.set_xlabel("Margin")
    ax.set_ylabel("Count")
    ax.set_title("Score Margins")
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PPC summary to {save_path}")
    
    plt.close(fig)


def plot_sensitivity_results(
    results: List[Dict],
    param_name: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Plot sensitivity analysis results for one parameter.
    """
    plt, _ = _import_matplotlib()
    
    param_values = [r["param_value"] for r in results]
    elbos = [r["elbo"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    cover_at_2s = [r["cover_at_2"] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # ELBO vs param
    ax = axes[0]
    ax.plot(param_values, elbos, 'o-', color='steelblue')
    ax.set_xlabel(param_name)
    ax.set_ylabel("ELBO")
    ax.set_title(f"ELBO vs {param_name}")
    ax.grid(True, alpha=0.3)
    
    # Accuracy vs param
    ax = axes[1]
    ax.plot(param_values, accuracies, 'o-', color='green')
    ax.set_xlabel(param_name)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs {param_name}")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Cover@2 vs param
    ax = axes[2]
    ax.plot(param_values, cover_at_2s, 'o-', color='orange')
    ax.set_xlabel(param_name)
    ax.set_ylabel("Cover@2")
    ax.set_title(f"Cover@2 vs {param_name}")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved sensitivity plot to {save_path}")
    
    plt.close(fig)
