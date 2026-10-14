"""Fixed bins from the calibration data and smoothed histograms (D-05)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def calibration_edges(values: np.ndarray, n_bins: int, percentiles: Sequence[float]) -> np.ndarray:
    """Inner edges: `n_bins` equal bins between the two percentiles of the calibration values.

    `histogram` adds one overflow bin below and one above, so values outside the calibration
    range still count.
    """
    v = _finite(values)
    if v.size == 0:
        raise ValueError("no calibration values")
    lo, hi = np.percentile(v, percentiles)
    if hi <= lo:
        hi = np.nextafter(lo, np.inf)
    return np.linspace(lo, hi, n_bins + 1)


def bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin of each value: 0 below edges[0], len(edges) above edges[-1], inner bins in between."""
    idx = np.searchsorted(edges, values, side="right")
    idx[values == edges[-1]] = len(edges) - 1
    return idx


def histogram(values: np.ndarray, edges: np.ndarray, eps: float) -> np.ndarray:
    """Probability over len(edges) + 1 bins, with `eps` added to every bin before normalizing.

    NaN values are ignored. With no values the result is uniform.
    """
    v = _finite(values)
    counts = np.bincount(bin_index(v, edges), minlength=len(edges) + 1).astype(np.float64)
    p = counts + eps
    return p / p.sum()


def _finite(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    return v[~np.isnan(v)]
