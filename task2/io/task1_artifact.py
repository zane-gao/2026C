from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import numpy as np

from .dataset import build_dataset, build_masks, Dataset
from ..types import ArtifactBundle


def _extract_array(data: Dict, keys: Iterable[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _ensure_k_dim(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr[None, ...]
    return arr


def _pad_to_shape(arr: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    out = np.zeros(shape, dtype=arr.dtype)
    slices = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, shape))
    out[slices] = arr[slices]
    return out


def _theta_from_F(F: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    eps = 1e-8
    if F.ndim == 3:
        F = F[None, ...]
    K, S, T, N = F.shape
    theta = np.zeros_like(F)
    for k in range(K):
        for s in range(S):
            for t in range(T):
                mask = active_mask[s, t] if active_mask.ndim == 3 else np.ones((N,), dtype=bool)
                if mask.sum() == 0:
                    continue
                f = np.clip(F[k, s, t], eps, None)
                logf = np.log(f)
                mean = logf[mask].mean()
                theta[k, s, t] = logf - mean
    return theta


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _resolve_artifact_path(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    raise FileNotFoundError(path)


def load_task1_artifact(path: str, dataset: Optional[Dataset] = None, max_week: Optional[int] = None) -> ArtifactBundle:
    p = _resolve_artifact_path(path)

    data: Dict[str, np.ndarray] = {}

    if p.is_dir():
        candidates = [
            p / "task1_artifact.npz",
            p / "artifact.npz",
            p / "posterior_summary.npz",
            p / "theta_samples.npy",
            p / "theta.npy",
            p / "theta_mean.npy",
        ]
        for cand in candidates:
            if cand.exists():
                if cand.suffix == ".npz":
                    data = _load_npz(cand)
                elif cand.suffix == ".npy":
                    data = {"theta": np.load(cand)}
                break
    else:
        if p.suffix == ".npz":
            data = _load_npz(p)
        elif p.suffix == ".npy":
            data = {"theta": np.load(p)}

    if dataset is None:
        if max_week is None:
            max_week = 11
        dataset = build_dataset("2026_MCM_Problem_C_Data.csv", max_week=max_week)

    masks = build_masks(dataset)
    active_mask = masks["active_mask"]
    valid_mask = masks["valid_mask"]
    withdraw_mask = masks["withdraw_mask"]
    multi_elim_count = masks["multi_elim_count"]
    elim_real = masks["elim_mask"]
    active_init = masks["active_init"]

    theta = _extract_array(data, ["theta", "theta_samples", "theta_posterior"])
    F = _extract_array(data, ["F", "fan_share", "F_mean"])
    mu = _extract_array(data, ["mu", "mu_samples"])
    gamma = _extract_array(data, ["gamma", "gamma_samples"])
    epsilon = _extract_array(data, ["epsilon", "eps", "resid"])

    vm = _extract_array(data, ["valid_mask", "Valid"])
    if vm is not None:
        valid_mask = vm
    wm = _extract_array(data, ["withdraw_mask", "withdrew"])
    if wm is not None:
        withdraw_mask = wm
    dm = _extract_array(data, ["multi_elim_count", "d"])
    if dm is not None:
        multi_elim_count = dm
    am = _extract_array(data, ["active_init", "A_init", "A_s1"])
    if am is not None:
        active_init = am
    em = _extract_array(data, ["elim_real", "elim_mask"])
    if em is not None:
        elim_real = em

    if theta is None and F is not None:
        theta = _theta_from_F(F, active_mask)
    if theta is None:
        raise ValueError("Task1 artifact missing theta/F samples")

    theta = _ensure_k_dim(theta)
    if mu is not None:
        mu = _ensure_k_dim(mu) if mu.ndim >= 3 else mu
    if epsilon is not None:
        epsilon = _ensure_k_dim(epsilon)

    # align shapes to dataset
    S, T, N = dataset.J_obs.shape
    K_raw = theta.shape[0]
    
    # If only a single sample (point estimate), bootstrap to generate K samples
    # This is essential for proper uncertainty quantification
    if K_raw == 1:
        print("[WARN] Task1 artifact has only 1 sample (point estimate). Generating synthetic samples for uncertainty.")
        # Extract F_std if available for proper noise scale
        F_std = _extract_array(data, ["F_std", "F_stderr"])
        theta_base = theta[0]  # [S, T, N]
        # Default K for bootstrap if not specified
        K_bootstrap = 200
        if F_std is not None:
            # Use normal perturbation based on estimated std
            # Transform std from F space to theta space: std_theta ≈ std_F / F (approx)
            F_base = _extract_array(data, ["F_mean", "F"])
            if F_base is not None and F_base.ndim == 3:
                F_base = np.clip(F_base, 1e-8, None)
                theta_std = F_std / F_base  # approximate std in log space
                theta_std = np.clip(theta_std, 0.01, 2.0)  # prevent extreme values
            else:
                theta_std = np.full_like(theta_base, 0.1)
            rng = np.random.default_rng(42)
            theta = theta_base[None, ...] + rng.normal(0, theta_std[None, ...], size=(K_bootstrap, S, T, N))
        else:
            # Simple bootstrap with small perturbation
            rng = np.random.default_rng(42)
            noise_scale = 0.1  # default noise in theta space
            theta = theta_base[None, ...] + rng.normal(0, noise_scale, size=(K_bootstrap, S, T, N))
        print(f"  Generated {K_bootstrap} synthetic samples from point estimate.")
    
    K = theta.shape[0]
    theta = _pad_to_shape(theta, (K, S, T, N))
    if isinstance(valid_mask, np.ndarray) and valid_mask.shape != (S, T):
        valid_mask = _pad_to_shape(valid_mask.astype(bool), (S, T)).astype(bool)
    if isinstance(withdraw_mask, np.ndarray) and withdraw_mask.shape != (S, T, N):
        withdraw_mask = _pad_to_shape(withdraw_mask.astype(bool), (S, T, N)).astype(bool)
    if isinstance(multi_elim_count, np.ndarray) and multi_elim_count.shape != (S, T):
        multi_elim_count = _pad_to_shape(multi_elim_count.astype(int), (S, T)).astype(int)
    if isinstance(active_init, np.ndarray) and active_init.shape != (S, N):
        active_init = _pad_to_shape(active_init.astype(bool), (S, N)).astype(bool)
    if isinstance(elim_real, np.ndarray) and elim_real.shape != (S, T, N):
        elim_real = _pad_to_shape(elim_real.astype(bool), (S, T, N)).astype(bool)

    return ArtifactBundle(
        theta=theta,
        valid_mask=valid_mask.astype(bool),
        active_init=active_init.astype(bool),
        withdraw_mask=withdraw_mask.astype(bool),
        multi_elim_count=multi_elim_count.astype(int),
        season_ids=dataset.seasons,
        couple_names=dataset.couple_names,
        elim_real=elim_real.astype(bool) if isinstance(elim_real, np.ndarray) else None,
        mu=mu,
        gamma=gamma,
        epsilon=epsilon,
        rule_id=None,
    )
