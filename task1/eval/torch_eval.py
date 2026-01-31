"""
PyTorch model evaluation and posterior analysis.
Generates all required deliverables for Task1.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

from ..types import WeekObs, RuleParams, TensorPack


def softmax(x: np.ndarray, mask: np.ndarray, axis: int = -1) -> np.ndarray:
    """Masked softmax in numpy."""
    x = np.where(mask > 0, x, -1e9)
    x = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x) * mask
    return exp_x / (exp_x.sum(axis=axis, keepdims=True) + 1e-12)


def compute_fan_share(theta: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    """
    Convert theta to fan share F using softmax.
    
    Args:
        theta: [S, T, N] latent ability
        active_mask: [S, T, N] active indicators
    
    Returns:
        F: [S, T, N] fan share (simplex per active set)
    """
    return softmax(theta, active_mask, axis=-1)


def sample_posterior(model, n_samples: int = 100) -> Dict[str, np.ndarray]:
    """
    Sample from the variational posterior.
    
    Returns dict with:
        - theta: [n_samples, S, T, N]
        - F: [n_samples, S, T, N]
        - beta, gamma, etc.
    """
    model.eval()
    samples = {"theta": [], "F": []}
    
    with torch.no_grad():
        for _ in range(n_samples):
            latents = model.sample_latents(n_samples=1)
            theta = model.compute_theta(latents).squeeze(0).cpu().numpy()  # [S, T, N]
            samples["theta"].append(theta)
            
            # Compute F
            active_mask = model.active_mask.cpu().numpy()  # [S, T, N]
            F = compute_fan_share(theta, active_mask)
            samples["F"].append(F)
    
    return {
        "theta": np.stack(samples["theta"], axis=0),  # [n_samples, S, T, N]
        "F": np.stack(samples["F"], axis=0),
    }


def compute_posterior_summary(samples: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute posterior summary statistics.
    
    Returns:
        - F_mean, F_median, F_q05, F_q95, F_q025, F_q975
        - CI_width_90, CI_width_95
        - entropy per (s, t)
    """
    F = samples["F"]  # [n_samples, S, T, N]
    
    summary = {
        "F_mean": np.mean(F, axis=0),
        "F_median": np.median(F, axis=0),
        "F_q05": np.percentile(F, 5, axis=0),
        "F_q95": np.percentile(F, 95, axis=0),
        "F_q025": np.percentile(F, 2.5, axis=0),
        "F_q975": np.percentile(F, 97.5, axis=0),
        "F_std": np.std(F, axis=0),
    }
    
    # CI widths
    summary["CI_width_90"] = summary["F_q95"] - summary["F_q05"]
    summary["CI_width_95"] = summary["F_q975"] - summary["F_q025"]
    
    # Entropy per (s, t) - averaged over samples
    # H = -sum(F * log(F))
    F_safe = np.clip(F, 1e-12, 1.0)
    entropy = -np.sum(F_safe * np.log(F_safe), axis=-1)  # [n_samples, S, T]
    summary["entropy_mean"] = np.mean(entropy, axis=0)  # [S, T]
    summary["entropy_std"] = np.std(entropy, axis=0)
    
    return summary


def replay_elimination(
    F: np.ndarray,  # [S, T, N]
    tensors: TensorPack,
    week_obs: List[WeekObs],
    params: RuleParams,
) -> Dict[str, any]:
    """
    Replay eliminations using posterior mean F and compute metrics.
    
    Returns:
        - predicted_elim: list of predicted elimination indices
        - accuracy: elimination hit rate
        - cover_at_2: rate of true elim in predicted bottom 2
        - margins: score margins
    """
    from ..rules.utils import masked_softmax as np_softmax
    
    season_idx = {s: i for i, s in enumerate(tensors.seasons)}
    
    results = []
    correct = 0
    cover_2 = 0
    total_valid = 0
    
    for obs in week_obs:
        if not obs.valid:
            continue
        
        s_idx = season_idx[obs.season]
        t_idx = obs.week - 1
        
        F_w = F[s_idx, t_idx, :]  # [N]
        active = obs.active_mask.astype(float)
        
        # True elimination
        true_elim_idx = np.where(obs.elim_mask > 0)[0]
        if len(true_elim_idx) == 0:
            continue
        
        true_elim = true_elim_idx[0]  # Take first if multiple
        total_valid += 1
        
        # Compute combined score based on rule
        if obs.rule.value == "percent":
            # C = J_pct + F
            C = obs.J_pct + F_w
            # Predicted elim = argmin(C) among active
            C_masked = np.where(active > 0, C, 1e9)
            pred_elim = np.argmin(C_masked)
            
            # Bottom 2
            sorted_idx = np.argsort(C_masked)
            bottom_2 = sorted_idx[:2]
            
            # Margin
            C_sorted = np.sort(C_masked[active > 0])
            margin = C_sorted[1] - C_sorted[0] if len(C_sorted) > 1 else 0
            
        elif obs.rule.value == "rank":
            # Use theta (or J_z + F as proxy for goodness)
            goodness = obs.J_z + F_w
            badness = -goodness
            badness_masked = np.where(active > 0, badness, -1e9)
            pred_elim = np.argmax(badness_masked)
            
            sorted_idx = np.argsort(-badness_masked)  # worst first
            bottom_2 = sorted_idx[:2]
            
            badness_sorted = np.sort(badness_masked[active > 0])[::-1]
            margin = badness_sorted[0] - badness_sorted[1] if len(badness_sorted) > 1 else 0
            
        else:  # bottom2_save
            goodness = obs.J_z + F_w
            badness = -goodness
            badness_masked = np.where(active > 0, badness, -1e9)
            pred_elim = np.argmax(badness_masked)
            
            sorted_idx = np.argsort(-badness_masked)
            bottom_2 = sorted_idx[:2]
            
            badness_sorted = np.sort(badness_masked[active > 0])[::-1]
            margin = badness_sorted[0] - badness_sorted[1] if len(badness_sorted) > 1 else 0
        
        # Check accuracy
        if pred_elim == true_elim:
            correct += 1
        
        # Check cover@2
        if true_elim in bottom_2:
            cover_2 += 1
        
        results.append({
            "season": obs.season,
            "week": obs.week,
            "rule": obs.rule.value,
            "true_elim": int(true_elim),
            "pred_elim": int(pred_elim),
            "correct": pred_elim == true_elim,
            "in_bottom_2": true_elim in bottom_2,
            "margin": float(margin),
        })
    
    metrics = {
        "accuracy": correct / total_valid if total_valid > 0 else 0,
        "cover_at_2": cover_2 / total_valid if total_valid > 0 else 0,
        "total_valid_weeks": total_valid,
        "correct_predictions": correct,
    }
    
    return {"metrics": metrics, "details": results}


def compute_feasible_rate(
    samples: Dict[str, np.ndarray],
    tensors: TensorPack,
    week_obs: List[WeekObs],
    params: RuleParams,
    epsilon: float = 0.005,
) -> Dict[str, float]:
    """
    Compute the rate at which posterior samples satisfy the elimination constraints.
    
    For percent rule: C_elim <= C_survivor - epsilon for all survivors
    """
    F_samples = samples["F"]  # [n_samples, S, T, N]
    n_samples = F_samples.shape[0]
    season_idx = {s: i for i, s in enumerate(tensors.seasons)}
    
    feasible_counts = []
    
    for obs in week_obs:
        if not obs.valid:
            continue
        
        s_idx = season_idx[obs.season]
        t_idx = obs.week - 1
        
        elim_idx = np.where(obs.elim_mask > 0)[0]
        if len(elim_idx) == 0:
            continue
        
        active = obs.active_mask.astype(float)
        survivor_idx = np.where((obs.elim_mask == 0) & (active > 0))[0]
        
        if len(survivor_idx) == 0:
            continue
        
        feasible = 0
        for k in range(n_samples):
            F_w = F_samples[k, s_idx, t_idx, :]
            
            if obs.rule.value == "percent":
                C = obs.J_pct + F_w
                C_elim = C[elim_idx[0]]
                C_survivors = C[survivor_idx]
                
                # Check if C_elim <= C_survivor - epsilon for all survivors
                if np.all(C_elim <= C_survivors - epsilon):
                    feasible += 1
            else:
                # For rank/bottom2, check if elim has worst combined rank
                goodness = obs.J_z + F_w
                elim_goodness = goodness[elim_idx[0]]
                survivor_goodness = goodness[survivor_idx]
                
                if np.all(elim_goodness <= survivor_goodness):
                    feasible += 1
        
        feasible_counts.append(feasible / n_samples)
    
    return {
        "mean_feasible_rate": np.mean(feasible_counts) if feasible_counts else 0,
        "min_feasible_rate": np.min(feasible_counts) if feasible_counts else 0,
        "per_week_feasible_rates": feasible_counts,
    }


def generate_fan_share_table(
    summary: Dict[str, np.ndarray],
    tensors: TensorPack,
    week_obs: List[WeekObs],
) -> List[Dict]:
    """
    Generate a table of fan share estimates for all (season, week, couple).
    """
    season_idx = {s: i for i, s in enumerate(tensors.seasons)}
    
    rows = []
    for obs in week_obs:
        s_idx = season_idx[obs.season]
        t_idx = obs.week - 1
        
        active_idx = np.where(obs.active_mask > 0)[0]
        
        for i in active_idx:
            couple_name = tensors.couple_names[s_idx][i] if i < len(tensors.couple_names[s_idx]) else f"Couple_{i}"
            
            rows.append({
                "season": obs.season,
                "week": obs.week,
                "couple_idx": int(i),
                "couple_name": couple_name,
                "F_mean": float(summary["F_mean"][s_idx, t_idx, i]),
                "F_median": float(summary["F_median"][s_idx, t_idx, i]),
                "F_q05": float(summary["F_q05"][s_idx, t_idx, i]),
                "F_q95": float(summary["F_q95"][s_idx, t_idx, i]),
                "F_q025": float(summary["F_q025"][s_idx, t_idx, i]),
                "F_q975": float(summary["F_q975"][s_idx, t_idx, i]),
                "CI_width_90": float(summary["CI_width_90"][s_idx, t_idx, i]),
                "CI_width_95": float(summary["CI_width_95"][s_idx, t_idx, i]),
                "is_eliminated": bool(obs.elim_mask[i] > 0),
            })
    
    return rows
