import argparse
from datetime import datetime
from pathlib import Path
import numpy as np

import sys
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task2.config import load_config
from task2.io.dataset import build_dataset, build_masks
from task2.io.task1_artifact import load_task1_artifact
from task2.skill.fit_linear import fit_linear_skill
from task2.data.calibrate import compute_jmax, match_lambda_for_std
from task2.skill.generate import generate_linear_samples
from task2.engine.crn import make_crn
from task2.engine.simulate import simulate_trajectories
from task2.io.export import save_npz, save_json, save_csv, hash_dict


def _load_skill_params(cfg, dataset, masks):
    cache_dir = Path(cfg.paths.output_dir) / "task2" / "skill_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_hash = hash_dict(cfg.to_dict())
    cache_path = cache_dir / f"skill_params_{run_hash}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return {
            "a": data["a"],
            "b": data["b"],
            "alpha": data["alpha"],
            "sigma_J": float(data["sigma_J"][0]) if data["sigma_J"].size else 1.0,
            "J_max": data["J_max"],
            "lambda_st": data["lambda_st"],
        }
    fit = fit_linear_skill(dataset.J_obs, masks["active_mask"], masks["valid_mask"])
    a = fit["a"]
    b = fit["b"]
    alpha = fit["alpha"]
    sigma_J = float(fit["sigma_J"]) if cfg.skill.sigma_J <= 0 else float(cfg.skill.sigma_J)
    J_max = compute_jmax(dataset.J_obs, masks["active_mask"], policy=cfg.skill.J_max_policy)
    t_idx = (np.arange(dataset.T_max) + 1).astype(float)
    base = a[:, None, :] + b[:, None, :] * t_idx[None, :, None]
    tildeJ = alpha[:, :, None] + base
    target_std = np.zeros_like(J_max)
    for s in range(dataset.J_obs.shape[0]):
        for t in range(dataset.J_obs.shape[1]):
            vals = dataset.J_obs[s, t][masks["active_mask"][s, t]]
            target_std[s, t] = float(np.std(vals)) if vals.size > 0 else 0.0
    lambda_st = match_lambda_for_std(
        tildeJ,
        J_max,
        target_std,
        masks["active_mask"],
        lam_min=cfg.skill.lambda_min,
        lam_max=cfg.skill.lambda_max,
        iters=cfg.skill.lambda_iters,
    ) if cfg.skill.std_match else np.ones_like(J_max)

    save_npz(cache_path, a=a, b=b, alpha=alpha, sigma_J=np.array([sigma_J]), J_max=J_max, lambda_st=lambda_st)
    return {"a": a, "b": b, "alpha": alpha, "sigma_J": sigma_J, "J_max": J_max, "lambda_st": lambda_st}


def _write_week_stats(out_path: Path, margin: np.ndarray, season_ids):
    rows = []
    K, S, T = margin.shape
    for s in range(S):
        for t in range(T):
            vals = margin[:, s, t]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                mean = q05 = q95 = float('nan')
            else:
                mean = float(vals.mean())
                q05 = float(np.quantile(vals, 0.05))
                q95 = float(np.quantile(vals, 0.95))
            rows.append({
                "season": season_ids[s],
                "week": t + 1,
                "margin_mean": mean,
                "margin_q05": q05,
                "margin_q95": q95,
            })
    save_csv(out_path, rows, fieldnames=["season", "week", "margin_mean", "margin_q05", "margin_q95"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Task2: run simulation")
    parser.add_argument("--config", type=str, required=False, help="Config path")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config(None)
    dataset = build_dataset(cfg.paths.data_csv, max_week=cfg.simulation.max_week)
    masks = build_masks(dataset)
    artifact = load_task1_artifact(cfg.paths.task1_artifact, dataset=dataset, max_week=cfg.simulation.max_week)

    missing_mode_b = artifact.mu is None or artifact.gamma is None or artifact.epsilon is None
    if missing_mode_b and "B" in cfg.simulation.modes:
        print("Mode B parameters missing; using Mode A preference for Mode B runs.")

    K = min(cfg.simulation.K, artifact.theta.shape[0])
    theta_samples = artifact.theta[:K]

    skill_params = _load_skill_params(cfg, dataset, masks)
    rng = np.random.default_rng(cfg.runtime.seed)
    S_samples, tildeJ, _ = generate_linear_samples(
        skill_params["a"],
        skill_params["b"],
        skill_params["alpha"],
        skill_params["sigma_J"],
        K,
        rng=rng,
    )

    crn = make_crn(cfg.runtime.seed, K, dataset.J_obs.shape[0], dataset.T_max, dataset.N_max)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path(cfg.paths.output_dir) / "task2" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", cfg.to_dict())

    for mode in cfg.simulation.modes:
        for mech in cfg.simulation.mechanisms:
            if mech == "W_MIX":
                for w in cfg.mixture.w_grid:
                    mix_cfg = {"kappa": cfg.mixture.kappa, "kappa_r": cfg.mixture.kappa_r, "w": float(w)}
                    traj = simulate_trajectories(
                        mode,
                        "W_MIX",
                        theta_samples,
                        artifact.mu,
                        artifact.gamma,
                        artifact.epsilon,
                        S_samples,
                        tildeJ,
                        artifact.valid_mask,
                        artifact.active_init,
                        artifact.withdraw_mask,
                        artifact.multi_elim_count,
                        skill_params["J_max"],
                        skill_params["lambda_st"],
                        cfg.simulation.tie_mode,
                        cfg.simulation.soft_percent,
                        cfg.simulation.soft_rank,
                        crn,
                        cfg.judges_save.__dict__,
                        mix_cfg,
                    )
                    traj["valid_mask"] = artifact.valid_mask
                    if artifact.elim_real is not None:
                        traj["elim_real"] = artifact.elim_real
                    traj["season_ids"] = np.array(artifact.season_ids)
                    tag = f"W_MIX_w{float(w):.2f}"
                    save_npz(out_dir / f"trajectories_{mode}_{tag}.npz", **traj)
                    _write_week_stats(out_dir / f"week_stats_{mode}_{tag}.csv", traj["margin"], artifact.season_ids)
            else:
                mix_cfg = {"kappa": cfg.mixture.kappa, "kappa_r": cfg.mixture.kappa_r, "w": 0.5}
                traj = simulate_trajectories(
                    mode,
                    mech,
                    theta_samples,
                    artifact.mu,
                    artifact.gamma,
                    artifact.epsilon,
                    S_samples,
                    tildeJ,
                    artifact.valid_mask,
                    artifact.active_init,
                    artifact.withdraw_mask,
                    artifact.multi_elim_count,
                    skill_params["J_max"],
                    skill_params["lambda_st"],
                    cfg.simulation.tie_mode,
                    cfg.simulation.soft_percent,
                    cfg.simulation.soft_rank,
                    crn,
                    cfg.judges_save.__dict__,
                    mix_cfg,
                )
                traj["valid_mask"] = artifact.valid_mask
                if artifact.elim_real is not None:
                    traj["elim_real"] = artifact.elim_real
                traj["season_ids"] = np.array(artifact.season_ids)
                save_npz(out_dir / f"trajectories_{mode}_{mech}.npz", **traj)
                _write_week_stats(out_dir / f"week_stats_{mode}_{mech}.csv", traj["margin"], artifact.season_ids)

    print(f"Saved simulation outputs to {out_dir}")


if __name__ == "__main__":
    main()
