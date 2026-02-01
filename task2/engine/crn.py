from __future__ import annotations

import numpy as np


def make_crn(seed: int, K: int, S: int, T: int, N: int) -> dict:
    ss = np.random.SeedSequence(seed)
    child = ss.spawn(K)
    tie_noise = np.zeros((K, S, T, N), dtype=float)
    gumbel_noise = np.zeros((K, S, T, N), dtype=float)
    save_u = np.zeros((K, S, T), dtype=float)
    for k in range(K):
        rng = np.random.default_rng(child[k])
        tie_noise[k] = rng.normal(size=(S, T, N))
        gumbel_noise[k] = rng.gumbel(size=(S, T, N))
        save_u[k] = rng.random(size=(S, T))
    return {
        "tie_noise": tie_noise,
        "gumbel_noise": gumbel_noise,
        "save_u": save_u,
    }
