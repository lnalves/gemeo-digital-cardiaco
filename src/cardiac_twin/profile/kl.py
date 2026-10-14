"""Kullback-Leibler divergence between smoothed histograms (TC Seção 2.3.4, D-05)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL(P ‖ Q) = Σ p log(p / q). Both must be positive and sum to 1 (see `histogram`)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("distributions with different bins")
    if np.any(p <= 0) or np.any(q <= 0):
        raise ValueError("distributions must be smoothed (no empty bins)")
    return float(np.sum(p * np.log(p / q)))


def kl_sum(p: Mapping[str, np.ndarray], q: Mapping[str, np.ndarray]) -> float:
    """D = Σ_f D_KL(P_f ‖ Q_f) over the features of the profile."""
    if set(p) != set(q):
        raise ValueError("different feature sets")
    return sum(kl(p[f], q[f]) for f in q)
