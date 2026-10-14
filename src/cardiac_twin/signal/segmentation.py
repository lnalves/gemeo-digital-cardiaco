"""Beat windows [R - pre, R + post), per-beat normalization and K-beat context."""

from __future__ import annotations

import numpy as np


def refine_r(x: np.ndarray, r_samples: np.ndarray, half: int) -> np.ndarray:
    """Move each R to the largest |x| within ±`half` samples of the filtered signal.

    Applied to the annotations (training) and to the detector output (streaming) alike, so both
    cut beat windows around the same point despite the phase delay of the causal filter (D-16).
    """
    x = np.asarray(x)
    out = np.asarray(r_samples, dtype=np.int64).copy()
    for k, r in enumerate(out):
        lo, hi = max(0, r - half), min(len(x), r + half + 1)
        out[k] = lo + int(np.argmax(np.abs(x[lo:hi])))
    return out


def valid_windows(r_samples: np.ndarray, n_samples: int, pre: int, post: int) -> np.ndarray:
    """Mask of beats whose whole window fits in the signal (edge beats are discarded)."""
    r = np.asarray(r_samples)
    return (r - pre >= 0) & (r + post <= n_samples)


def extract_beats(x: np.ndarray, r_samples: np.ndarray, pre: int, post: int) -> np.ndarray:
    """Return (n, pre + post) windows with the R sample at index `pre`. Every window must fit."""
    r = np.asarray(r_samples, dtype=np.int64)
    if not valid_windows(r, len(x), pre, post).all():
        raise ValueError("beat window outside the signal; filter with valid_windows first")
    idx = r[:, None] + np.arange(-pre, post)[None, :]
    return np.asarray(x)[idx]


def zscore(beats: np.ndarray) -> np.ndarray:
    """Normalize each beat (row) to zero mean and unit variance; a flat beat stays at zero."""
    beats = np.asarray(beats, dtype=np.float64)
    mean = beats.mean(axis=-1, keepdims=True)
    std = beats.std(axis=-1, keepdims=True)
    return (beats - mean) / np.where(std > np.finfo(np.float32).eps, std, 1.0)


def normalize(beats: np.ndarray, method: str) -> np.ndarray:
    if method == "zscore":
        return zscore(beats)
    raise ValueError(f"unknown normalization {method!r}")


def context_indices(n_beats: int, context: int, future: int) -> tuple[np.ndarray, np.ndarray]:
    """Beat indices of each K-beat sequence [i - (K-1-future), ..., i, ..., i + future].

    Return (centers (m,), sequences (m, K)); only centers with a full context are kept. Use per
    record: a sequence must never cross two records.
    """
    past = context - 1 - future
    if past < 0:
        raise ValueError("context must be larger than the number of future beats")
    centers = np.arange(past, n_beats - future)
    return centers, centers[:, None] + np.arange(-past, future + 1)[None, :]
