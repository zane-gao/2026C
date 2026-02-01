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
from task2.io.dataset import build_dataset
from task2.io.social_proxy import load_social_proxy_map, normalize_name, save_social_proxy_csv
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


def _summarize(values: np.ndarray, q_low: float = 0.05, q_high: float = 0.95) -> dict:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return {"mean": float('nan'), "q_low": float('nan'), "q_high": float('nan')}
    return {
        "mean": float(np.mean(vals)),
        "q_low": float(np.quantile(vals, q_low)),
        "q_high": float(np.quantile(vals, q_high)),
    }


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    n = len(values)
    ranks[order] = np.arange(1, n + 1)
    return ranks / n


def _split_couple_name(name: str) -> tuple[str, str]:
    if not name:
        return "", ""
    if " / " in name:
        parts = name.split(" / ", 1)
    elif "/" in name:
        parts = name.split("/", 1)
    else:
        return name, ""
    return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""


def _build_exo_arrays(dataset, proxy_map: dict):
    S = len(dataset.seasons)
    N = dataset.N_max
    P_cele = np.full((S, N), np.nan, dtype=float)
    P_partner = np.full((S, N), np.nan, dtype=float)
    missing_cele = np.ones((S, N), dtype=bool)
    missing_partner = np.ones((S, N), dtype=bool)
    for s_idx, season in enumerate(dataset.seasons):
        for i_idx, name in enumerate(dataset.couple_names[s_idx]):
            if not name:
                continue
            celeb, partner = _split_couple_name(name)
            key = (season, normalize_name(celeb), normalize_name(partner))
            rec = proxy_map.get(key)
            if rec is None:
                continue
            p_cele = rec.get("P_cele")
            p_partner = rec.get("P_partner")
            if p_cele is not None:
                P_cele[s_idx, i_idx] = float(p_cele)
            if p_partner is not None:
                P_partner[s_idx, i_idx] = float(p_partner)
            missing_cele[s_idx, i_idx] = bool(rec.get("missing_cele_total", True))
            missing_partner[s_idx, i_idx] = bool(rec.get("missing_partner_total", True))
    return P_cele, P_partner, missing_cele, missing_partner


def _compute_exo_group(S_bar: np.ndarray, P: np.ndarray, q: float):
    S_mean = S_bar.mean(axis=0)
    S, N = S_mean.shape
    group = np.zeros((S, N), dtype=bool)
    pS_full = np.full((S, N), np.nan, dtype=float)
    pP_full = np.full((S, N), np.nan, dtype=float)
    for s in range(S):
        mask = np.isfinite(S_mean[s]) & np.isfinite(P[s])
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        pS = _percentile_ranks(S_mean[s, idx])
        pP = _percentile_ranks(P[s, idx])
        pS_full[s, idx] = pS
        pP_full[s, idx] = pP
        ci = pP - pS
        cutoff = np.quantile(ci, 1.0 - q) if ci.size > 0 else 1.0
        group_idx = idx[ci >= cutoff]
        group[s, group_idx] = True
    return group, pS_full, pP_full


def _build_quadrant_groups(pS: np.ndarray, pP: np.ndarray, q: float):
    high_s = pS >= (1.0 - q)
    low_s = pS <= q
    high_p = pP >= (1.0 - q)
    low_p = pP <= q
    g1 = high_s & high_p
    g2 = high_s & low_p
    g3 = low_s & high_p
    g4 = low_s & low_p
    return {
        "G1_highS_highP": g1,
        "G2_highS_lowP": g2,
        "G3_lowS_highP": g3,
        "G4_lowS_lowP": g4,
    }


def _compute_exopop(final_rank: np.ndarray, P: np.ndarray) -> dict:
    vals = []
    K, S, N = final_rank.shape
    for k in range(K):
        for s in range(S):
            rank = final_rank[k, s]
            mask = (rank > 0) & np.isfinite(P[s])
            if mask.sum() < 2:
                continue
            vals.append(spearmanr(rank[mask], -P[s, mask]))
    return _summarize(np.array(vals))


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
    parser.add_argument("--social", type=str, default=None, help="Social proxy csv or enriched data file (xlsx/csv)")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        raise FileNotFoundError(str(run_dir))

    cfg = load_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    sim_cfg = cfg.get("simulation", {}) if isinstance(cfg, dict) else {}
    topk_jaccard = int(sim_cfg.get("topk_jaccard", 3))
    topk_final = int(sim_cfg.get("topk_final", 3))
    controversy_q = float(sim_cfg.get("controversy_q", 0.2))
    quad_q = float(sim_cfg.get("quad_q", 0.5))
    cap_m_list = [1, 2, 3]

    # load social proxy map if available
    social_map = {}
    social_source = None
    candidates = []
    if args.social:
        candidates.append(Path(args.social))
    candidates.append(run_dir / "social_proxy.csv")
    candidates.append(Path("outputs/task1/results/social_proxy.csv"))
    if isinstance(cfg, dict):
        paths_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
        data_path = paths_cfg.get("data_csv")
        if data_path:
            candidates.append(Path(data_path))
    for cand in candidates:
        if not cand or not cand.exists():
            continue
        social_map = load_social_proxy_map(str(cand))
        if social_map:
            social_source = str(cand)
            break

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

    # build exogenous popularity arrays if available
    P_cele = None
    P_partner = None
    missing_cele = None
    missing_partner = None
    if social_map:
        try:
            data_path = "2026_MCM_Problem_C_Data.csv"
            if isinstance(cfg, dict):
                paths_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
                data_path = paths_cfg.get("data_csv") or data_path
            dataset = build_dataset(data_path, max_week=sim_cfg.get("max_week", None))
            P_cele, P_partner, missing_cele, missing_partner = _build_exo_arrays(dataset, social_map)
            if social_source and not social_source.lower().endswith("social_proxy.csv"):
                save_social_proxy_csv(social_map, run_dir / "social_proxy.csv")
            report["social"] = {
                "source": social_source,
                "n_records": len(social_map),
            }
        except Exception as exc:
            print(f"[WARN] Failed to load social proxy: {exc}")
            social_map = {}

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
        if P_cele is not None:
            report["metrics"][mode][mech]["ExoPop"] = _compute_exopop(traj["final_rank"], P_cele)

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
        group_exo = None
        quad_groups = None
        if P_cele is not None:
            group_exo, pS_full, pP_full = _compute_exo_group(traj_r["S_bar"], P_cele, q=controversy_q)
            quad_groups = _build_quadrant_groups(pS_full, pP_full, quad_q)
            report["metrics"].setdefault("A", {})["ATE_group_exo"] = {
                "controversy_exo": compute_group_ate(ite, group_exo),
            }
            report["metrics"].setdefault("A", {})["ATE_group_quad"] = {
                name: compute_group_ate(ite, mask) for name, mask in quad_groups.items()
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
                in_exo = bool(group_exo[s, i]) if group_exo is not None else False
                quad_label = ""
                if quad_groups is not None:
                    for label, mask in quad_groups.items():
                        if mask[s, i]:
                            quad_label = label
                            break
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
                    "in_exo_controversy": in_exo,
                    "exo_quad": quad_label,
                    "P_cele": float(P_cele[s, i]) if P_cele is not None else float('nan'),
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
                "in_exo_controversy",
                "exo_quad",
                "P_cele",
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
