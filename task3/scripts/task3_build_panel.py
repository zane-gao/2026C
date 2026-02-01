from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import json

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task3.config import load_config, Config
from task3.io.dataset import build_dataset, build_masks, find_data_csv
from task3.io.task1_artifact import load_task1_artifact
from task3.io.export import save_parquet
from task3.data.panel import build_panel, compute_ref_couples


def _make_run_id() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task3: build panel")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--social", type=str, default=None)
    parser.add_argument("--task1", type=str, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data:
        cfg.paths.data_csv = args.data
    if args.social:
        cfg.paths.enriched_path = args.social
    if args.task1:
        cfg.paths.task1_artifact = args.task1
    if args.k:
        cfg.runtime.k_subsample = args.k
    if args.output:
        cfg.paths.output_root = args.output

    data_csv = find_data_csv(cfg.paths.data_csv)
    dataset = build_dataset(data_csv, max_week=cfg.panel.max_week)
    artifact = load_task1_artifact(cfg.paths.task1_artifact, dataset=dataset, max_week=cfg.panel.max_week)

    masks = build_masks(dataset)
    active_mask = masks["active_mask"]
    valid_mask = artifact.valid_mask if hasattr(artifact, "valid_mask") else masks["valid_mask"]
    withdraw_mask = artifact.withdraw_mask if hasattr(artifact, "withdraw_mask") else masks["withdraw_mask"]
    elim_mask = masks["elim_mask"]

    ref_idx = compute_ref_couples(artifact.theta, active_mask)
    panel = build_panel(
        dataset,
        data_csv=data_csv,
        social_path=cfg.paths.enriched_path,
        active_mask=active_mask,
        valid_mask=valid_mask,
        elim_mask=elim_mask,
        withdraw_mask=withdraw_mask,
        ref_idx=ref_idx,
    )

    # k subsample
    K = artifact.theta.shape[0]
    k_sub = cfg.runtime.k_subsample
    if k_sub is None or k_sub <= 0 or k_sub > K:
        k_index = list(range(K))
    else:
        import numpy as np
        rng = np.random.default_rng(cfg.runtime.seed)
        k_index = sorted(rng.choice(K, size=k_sub, replace=False).tolist())

    run_id = args.run_id or _make_run_id()
    run_dir = Path(cfg.paths.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_parquet(run_dir / "panel.parquet", panel)
    (run_dir / "k_index.json").write_text(json.dumps(k_index, indent=2), encoding="utf-8")
    cfg.save(run_dir / "config.resolved.yaml")

    print(f"Saved panel to {run_dir}")


if __name__ == "__main__":
    main()
