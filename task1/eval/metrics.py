from __future__ import annotations

import numpy as np


def acc(pred_elim, true_elim):
    if pred_elim is None or true_elim is None:
        return None
    return 1.0 if pred_elim == true_elim else 0.0


def cover_at_2(pred_bottom2, true_elim):
    if true_elim is None:
        return None
    return 1.0 if true_elim in pred_bottom2 else 0.0


def margin_percent(C):
    # assumes C is array for active contestants
    order = np.argsort(C)
    if len(order) < 2:
        return 0.0
    return float(C[order[1]] - C[order[0]])
