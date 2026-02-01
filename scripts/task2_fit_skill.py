import argparse
from pathlib import Path
import numpy as np

import sys
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task2.config import load_config
from task2.io.dataset import build_dataset, build_masks
from task2.skill.fit_linear import fit_linear_skill
from task2.data.calibrate import compute_jmax, match_lambda_for_std
from task2.io.export import save_npz, hash_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Task2: fit skill process and calibration")
    parser.add_argument("--config", type=str, required=False, help="Config path (json/yaml)")
    parser.add_argument("--out", type=str, default=None, help="Output npz path (optional)")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config(None)
    dataset = build_dataset(cfg.paths.data_csv, max_week=cfg.simulation.max_week)
    masks = build_masks(dataset)

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

    params = {
        "a": a,
        "b": b,
        "alpha": alpha,
        "sigma_J": np.array([sigma_J], dtype=float),
        "J_max": J_max,
        "lambda_st": lambda_st,
    }

    out_path = args.out
    if out_path is None:
        cache_dir = Path(cfg.paths.output_dir) / "task2" / "skill_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        run_hash = hash_dict(cfg.to_dict())
        out_path = str(cache_dir / f"skill_params_{run_hash}.npz")

    save_npz(out_path, **params)
    print(f"Saved skill params to {out_path}")


if __name__ == "__main__":
    main()
