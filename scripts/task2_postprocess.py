from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from typing import Optional
import sys
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task2.io.export import save_json, save_csv, load_json
from task2.eval.metrics_core import compute_metrics, compute_consistency, spearmanr
from task2.eval.causal import compute_ite, compute_group_ate
from task2.eval.controversy import compute_controversy_group
from task2.viz import fig_pipeline, fig_parallel_universe, fig_rank_bar, fig_survival_ite_ate, fig_margin, fig_bias_box
from task2.viz import fig_js_scan, fig_radar_pareto, fig_entropy_div, fig_weight_sensitivity


def _load_traj(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _compute_ir(traj: dict, group_mask: np.ndarray) -> float:
    if "bottom2_mask" not in traj or "saved_mask" not in traj:
        return float('nan')
    bottom2 = traj["bottom2_mask"]
    saved = traj["saved_mask"]
    if bottom2 is None or saved is None:
        return float('nan')
    K, S, T, N = bottom2.shape
    num = 0.0
    den = 0.0
    for s in range(S):
        for i in range(N):
            if not group_mask[s, i]:
                continue
            b = bottom2[:, s, :, i]
            s_mask = saved[:, s, :, i]
            den += b.sum()
            num += (b & s_mask).sum()
    return float(num / den) if den > 0 else float('nan')


def _top_group(values: np.ndarray, q: float) -> np.ndarray:
    mean = values.mean(axis=0)
    S, N = mean.shape
    group = np.zeros((S, N), dtype=bool)
    for s in range(S):
        mask = np.isfinite(mean[s])
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        cutoff = np.quantile(mean[s, idx], 1.0 - q)
        group_idx = idx[mean[s, idx] >= cutoff]
        group[s, group_idx] = True
    return group


def _season_divergence(elim_matrix: np.ndarray, elim_real: Optional[np.ndarray],
                       valid_mask: np.ndarray, s_idx: int) -> float:
    if elim_real is None:
        return float('nan')
    K, S, T, N = elim_matrix.shape
    diffs = []
    for k in range(K):
        for t in range(T):
            if not valid_mask[s_idx, t]:
                continue
            real = set(np.where(elim_real[s_idx, t])[0].tolist())
            sim = set(np.where(elim_matrix[k, s_idx, t])[0].tolist())
            diffs.append(1.0 if real != sim else 0.0)
    return float(np.mean(diffs)) if diffs else float('nan')


def _consensus_topk(final_rank: np.ndarray, topk_final: int) -> tuple[tuple[int, ...], float]:
    counts: dict[tuple[int, ...], int] = {}
    total = 0
    for k in range(final_rank.shape[0]):
        rank = final_rank[k]
        mask = rank > 0
        if mask.sum() == 0:
            continue
        k_final = min(topk_final, int(mask.sum()))
        topk = tuple(sorted(np.where(mask)[0][np.argsort(rank[mask])][:k_final]))
        counts[topk] = counts.get(topk, 0) + 1
        total += 1
    if total == 0:
        return tuple(), float('nan')
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0], float(best[1] / total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task2: postprocess metrics and plots")
    parser.add_argument("--run", type=str, required=True, help="Run directory (outputs/task2/run_xxx)")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        raise FileNotFoundError(str(run_dir))

    cfg = load_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    sim_cfg = cfg.get("simulation", {}) if isinstance(cfg, dict) else {}
    topk_jaccard = int(sim_cfg.get("topk_jaccard", 3))
    topk_final = int(sim_cfg.get("topk_final", 3))
    controversy_q = float(sim_cfg.get("controversy_q", 0.2))
    cap_m_list = [1, 2, 3]

    traj_map = {}
    for path in run_dir.glob("trajectories_*.npz"):
        name = path.stem.replace("trajectories_", "")
        parts = name.split("_", 1)
        if len(parts) != 2:
            continue
        mode, mech = parts[0], parts[1]
        traj_map[(mode, mech)] = _load_traj(path)

    if not traj_map:
        raise RuntimeError("No trajectories found in run directory")

    report = {"metrics": {}}

    # season summary
    season_rows = []
    for (mode, mech), traj in traj_map.items():
        winner = traj["winner"]
        final_rank = traj["final_rank"]
        S_bar = traj["S_bar"]
        F_bar = traj["F_bar"]
        season_ids = traj.get("season_ids", None)
        T = traj["margin"].shape[2]
        valid_mask = traj.get("valid_mask", np.ones((winner.shape[1], T), dtype=bool))
        elim_real = traj.get("elim_real", None)
        K, S = winner.shape
        N = final_rank.shape[2]
        for s in range(S):
            winners = winner[:, s]
            valid = winners[winners >= 0]
            if valid.size == 0:
                top_idx = -1
                prob = float('nan')
            else:
                counts = np.bincount(valid.astype(int), minlength=N)
                top_idx = int(np.argmax(counts))
                prob = float(counts[top_idx] / counts.sum()) if counts.sum() > 0 else 0.0
            topk_set, topk_prob = _consensus_topk(final_rank[:, s], topk_final)
            topk_str = "|".join(str(i) for i in topk_set)

            spearman_skill_vals = []
            spearman_pop_vals = []
            for k in range(K):
                rank = final_rank[k, s]
                mask = rank > 0
                if mask.sum() < 2:
                    continue
                spearman_skill_vals.append(spearmanr(rank[mask], -S_bar[k, s, mask]))
                spearman_pop_vals.append(spearmanr(rank[mask], -F_bar[k, s, mask]))
            spearman_skill_vals = [v for v in spearman_skill_vals if np.isfinite(v)]
            spearman_pop_vals = [v for v in spearman_pop_vals if np.isfinite(v)]
            spearman_skill = float(np.mean(spearman_skill_vals)) if spearman_skill_vals else float('nan')
            spearman_pop = float(np.mean(spearman_pop_vals)) if spearman_pop_vals else float('nan')
            div_mean = _season_divergence(traj["elim_matrix"], elim_real, valid_mask, s)

            season_rows.append({
                "season": int(season_ids[s]) if season_ids is not None else s + 1,
                "mode": mode,
                "mechanism": mech,
                "winner_idx": top_idx,
                "winner_prob": prob,
                "topk_consensus": topk_str,
                "topk_prob": topk_prob,
                "spearman_skill": spearman_skill,
                "spearman_pop": spearman_pop,
                "divergence": div_mean,
            })

    save_csv(
        run_dir / "season_summary.csv",
        season_rows,
        fieldnames=[
            "season",
            "mode",
            "mechanism",
            "winner_idx",
            "winner_prob",
            "topk_consensus",
            "topk_prob",
            "spearman_skill",
            "spearman_pop",
            "divergence",
        ],
    )

    # metrics per mechanism
    for (mode, mech), traj in traj_map.items():
        T = traj["margin"].shape[2]
        valid_mask = traj.get("valid_mask", np.ones((traj["winner"].shape[1], T), dtype=bool))
        elim_real = traj.get("elim_real", None)
        metrics = compute_metrics(
            traj,
            valid_mask=valid_mask,
            topk_final=topk_final,
            elim_real=elim_real,
            cap_m_list=cap_m_list,
        )
        report["metrics"].setdefault(mode, {})[mech] = metrics

    # consistency P vs R per mode
    for mode in set(k[0] for k in traj_map.keys()):
        key_p = (mode, "P")
        key_r = (mode, "R")
        if key_p in traj_map and key_r in traj_map:
            report["metrics"].setdefault(mode, {})["consistency_P_R"] = compute_consistency(
                traj_map[key_p], traj_map[key_r], topk_jaccard=topk_jaccard
            )

    # ITE/ATE for mode A if P and R present
    ite_rows = []
    if ("A", "P") in traj_map and ("A", "R") in traj_map:
        traj_r = traj_map[("A", "R")]
        season_ids = traj_r.get("season_ids", None)
        ite = compute_ite(traj_r, traj_map[("A", "P")])
        group_cont = compute_controversy_group(traj_r["S_bar"], traj_r["F_bar"], q=controversy_q)
        group_skill = _top_group(traj_r["S_bar"], controversy_q)
        group_pop = _top_group(traj_r["F_bar"], controversy_q)
        report["metrics"].setdefault("A", {})["ATE_group"] = {
            "controversy": compute_group_ate(ite, group_cont),
            "skill_top": compute_group_ate(ite, group_skill),
            "pop_top": compute_group_ate(ite, group_pop),
        }
        tau_T = ite["tau_T_mean"]
        tau_T_q05 = ite["tau_T_q05"]
        tau_T_q95 = ite["tau_T_q95"]
        tau_pi = ite["tau_pi_mean"]
        tau_pi_q05 = ite["tau_pi_q05"]
        tau_pi_q95 = ite["tau_pi_q95"]
        S, N = tau_T.shape
        for s in range(S):
            season_id = int(season_ids[s]) if season_ids is not None else s + 1
            for i in range(N):
                ite_rows.append({
                    "season": season_id,
                    "contestant": i,
                    "tau_T_mean": float(tau_T[s, i]),
                    "tau_T_q05": float(tau_T_q05[s, i]),
                    "tau_T_q95": float(tau_T_q95[s, i]),
                    "tau_pi_mean": float(tau_pi[s, i]),
                    "tau_pi_q05": float(tau_pi_q05[s, i]),
                    "tau_pi_q95": float(tau_pi_q95[s, i]),
                    "in_controversy": bool(group_cont[s, i]),
                    "in_skill_top": bool(group_skill[s, i]),
                    "in_pop_top": bool(group_pop[s, i]),
                })

    if ite_rows:
        save_csv(
            run_dir / "ite_ate.csv",
            ite_rows,
            fieldnames=[
                "season",
                "contestant",
                "tau_T_mean",
                "tau_T_q05",
                "tau_T_q95",
                "tau_pi_mean",
                "tau_pi_q05",
                "tau_pi_q95",
                "in_controversy",
                "in_skill_top",
                "in_pop_top",
            ],
        )

    # JS intercept rate
    for (mode, mech), traj in traj_map.items():
        if mech.startswith("JS"):
            group = compute_controversy_group(traj["S_bar"], traj["F_bar"], q=controversy_q)
            report["metrics"].setdefault(mode, {})["IR"] = _compute_ir(traj, group)

    save_json(run_dir / "task2_report.json", report)

    # placeholder plots - gracefully handle if matplotlib unavailable
    # Prepare data for plotting
    plot_data = {
        "traj_map": traj_map,
        "ite_rows": ite_rows,
        "report": report,
        "cfg": cfg
    }

    plot_funcs = [
        (fig_pipeline.plot, "fig01_pipeline.png"),
        (fig_parallel_universe.plot, "fig02_parallel_universe.png"),
        (fig_rank_bar.plot, "fig03_rank_bar.png"),
        (fig_survival_ite_ate.plot, "fig04_survival_ite_ate.png"),
        (fig_margin.plot, "fig05_margin.png"),
        (fig_bias_box.plot, "fig06_bias_box.png"),
        (fig_js_scan.plot, "fig07_js_scan.png"),
        (fig_radar_pareto.plot, "fig08_radar_pareto.png"),
        (fig_entropy_div.plot, "fig09_entropy_div.png"),
        (fig_weight_sensitivity.plot, "fig10_weight_sensitivity.png"),
    ]
    for plot_fn, filename in plot_funcs:
        try:
            plot_fn(run_dir / filename, data=plot_data)
        except Exception as e:
            print(f"[WARN] Failed to generate {filename}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Saved report and figures to {run_dir}")


if __name__ == "__main__":
    main()
