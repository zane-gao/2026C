from __future__ import annotations

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(values, dtype=float)
    i = 0
    rank = 1
    n = len(values)
    while i < n:
        j = i
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        avg = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        rank += (j - i)
        i = j
    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float('nan')
    rx = _rankdata(x)
    ry = _rankdata(y)
    vx = rx - rx.mean()
    vy = ry - ry.mean()
    denom = np.sqrt((vx ** 2).sum() * (vy ** 2).sum())
    if denom <= 0:
        return float('nan')
    return float((vx * vy).sum() / denom)


def jaccard_topk(rank_a: np.ndarray, rank_b: np.ndarray, k: int) -> float:
    if k <= 0:
        return float('nan')
    idx_a = np.where(rank_a > 0)[0]
    idx_b = np.where(rank_b > 0)[0]
    if idx_a.size == 0 or idx_b.size == 0:
        return float('nan')
    top_a = set(idx_a[np.argsort(rank_a[idx_a])][:k].tolist())
    top_b = set(idx_b[np.argsort(rank_b[idx_b])][:k].tolist())
    if not top_a and not top_b:
        return float('nan')
    return float(len(top_a & top_b) / len(top_a | top_b))


def entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return float('nan')
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def summarize(values: np.ndarray, q_low: float = 0.05, q_high: float = 0.95) -> dict:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return {"mean": float('nan'), "q_low": float('nan'), "q_high": float('nan')}
    return {
        "mean": float(np.mean(vals)),
        "q_low": float(np.quantile(vals, q_low)),
        "q_high": float(np.quantile(vals, q_high)),
    }


def compute_divergence(elim_matrix: np.ndarray, elim_real: np.ndarray, valid_mask: np.ndarray) -> float:
    if elim_real is None:
        return float('nan')
    K, S, T, N = elim_matrix.shape
    diffs = []
    for k in range(K):
        for s in range(S):
            for t in range(T):
                if not valid_mask[s, t]:
                    continue
                real = set(np.where(elim_real[s, t])[0].tolist())
                sim = set(np.where(elim_matrix[k, s, t])[0].tolist())
                diffs.append(1.0 if real != sim else 0.0)
    if not diffs:
        return float('nan')
    return float(np.mean(diffs))


def compute_metrics(traj: dict, valid_mask: np.ndarray, topk_final: int, elim_real: np.ndarray | None,
                    cap_m_list: list[int] | None = None) -> dict:
    winner = traj["winner"]
    final_rank = traj["final_rank"]
    S_bar = traj["S_bar"]
    F_bar = traj["F_bar"]
    margin = traj["margin"]
    K, S, N = final_rank.shape

    if cap_m_list is None:
        cap_m_list = [1, 2, 3]

    spearman_skill = []
    spearman_pop = []
    skill_win = []
    pop_win = []
    regret_s = []
    regret_f = []
    cap_win = {m: [] for m in cap_m_list}
    cap_final = {m: [] for m in cap_m_list}

    for k in range(K):
        for s in range(S):
            rank = final_rank[k, s]
            mask = rank > 0
            if mask.sum() < 2:
                continue
            spearman_skill.append(spearmanr(rank[mask], -S_bar[k, s, mask]))
            spearman_pop.append(spearmanr(rank[mask], -F_bar[k, s, mask]))
            w = winner[k, s]
            if w >= 0:
                skill_win.append(S_bar[k, s, w])
                pop_win.append(F_bar[k, s, w])
                regret_s.append(float(np.max(S_bar[k, s, mask]) - S_bar[k, s, w]))
                regret_f.append(float(np.max(F_bar[k, s, mask]) - F_bar[k, s, w]))
                order_skill = np.argsort(-S_bar[k, s, mask])
                top_skill_idx = np.where(mask)[0][order_skill]
                k_final = min(topk_final, mask.sum())
                top_final = np.where(mask)[0][np.argsort(rank[mask])][:k_final]
                for m in cap_m_list:
                    if top_skill_idx.size < m:
                        continue
                    top_m = top_skill_idx[:m]
                    cap_win[m].append(1.0 if w in top_m else 0.0)
                    if m > 0:
                        cap_final[m].append(float(len(set(top_m) & set(top_final)) / m))

    # winner entropy per season
    entropies = []
    for s in range(S):
        counts = np.bincount(winner[:, s][winner[:, s] >= 0], minlength=N)
        entropies.append(entropy_from_counts(counts))

    div = compute_divergence(traj["elim_matrix"], elim_real, valid_mask)

    # margin summary
    margin_summary = summarize(margin[np.isfinite(margin)])

    out = {
        "Merit": summarize(np.array(spearman_skill)),
        "Pop": summarize(np.array(spearman_pop)),
        "SkillWin": summarize(np.array(skill_win)),
        "PopWin": summarize(np.array(pop_win)),
        "H_win": summarize(np.array(entropies)),
        "Div": {"mean": div, "q_low": float('nan'), "q_high": float('nan')},
        "Regret_S": summarize(np.array(regret_s)),
        "Regret_F": summarize(np.array(regret_f)),
        "Margin": margin_summary,
    }
    for m in cap_m_list:
        out[f"Cap_win_{m}"] = summarize(np.array(cap_win[m]))
        out[f"Cap_final_{m}"] = summarize(np.array(cap_final[m]))
    return out


def compute_consistency(traj_p: dict, traj_r: dict, topk_jaccard: int) -> dict:
    winner_p = traj_p["winner"]
    winner_r = traj_r["winner"]
    final_p = traj_p["final_rank"]
    final_r = traj_r["final_rank"]
    K, S, N = final_p.shape
    wins = []
    jaccs = []
    rhos = []
    for k in range(K):
        for s in range(S):
            if winner_p[k, s] >= 0 and winner_r[k, s] >= 0:
                wins.append(1.0 if winner_p[k, s] == winner_r[k, s] else 0.0)
            # jaccard
            rank_p = final_p[k, s]
            rank_r = final_r[k, s]
            mask_p = rank_p > 0
            mask_r = rank_r > 0
            if mask_p.sum() > 0 and mask_r.sum() > 0:
                jaccs.append(jaccard_topk(rank_p, rank_r, topk_jaccard))
            # spearman
            mask = (rank_p > 0) & (rank_r > 0)
            if mask.sum() >= 2:
                rhos.append(spearmanr(rank_p[mask], rank_r[mask]))
    return {
        "W": summarize(np.array(wins)),
        "Jaccard": summarize(np.array(jaccs)),
        "Spearman": summarize(np.array(rhos)),
    }
