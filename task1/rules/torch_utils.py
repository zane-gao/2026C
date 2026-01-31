"""
PyTorch utilities for likelihood computation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute softmax over scores, masking out invalid entries."""
    # mask: 1 for valid, 0 for invalid
    scores = scores - scores.max()  # numerical stability
    exp_scores = torch.exp(scores) * mask
    return exp_scores / (exp_scores.sum() + 1e-12)


def soft_rank(x: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """Soft ranking using pairwise comparisons."""
    # For each element, count how many elements are smaller (soft)
    diff = x.unsqueeze(-1) - x.unsqueeze(-2)  # [N, N]
    return torch.sigmoid(diff / tau).sum(dim=-1)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid function."""
    return torch.sigmoid(x)
